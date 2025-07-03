import socket
import numpy as np
import struct
import pickle
import zlib
import cv2
import time
import threading
import queue
import logging

# ======================== 配置日志 ========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('tcp_transfer.log')
    ]
)
logger = logging.getLogger(__name__)

# ======================== 全局配置 ========================
HOST = '127.0.0.1'  # 服务器地址
PORT = 8888          # 服务器端口
# MAX_IMAGES = 5       # 最大图像数量
IMAGE_SIZE = (240, 240)  # 图像尺寸
RESPONSE_TIMEOUT = 10.0  # 响应超时时间（秒）

# ======================== 协议定义 ========================
CMD_HEADER = bytes.fromhex('eb9001')  # 指令+图像包头
IMG_HEADER = bytes.fromhex('eb9002')  # 仅图像包头
RESP_HEADER = bytes.fromhex('eb9003')  # 响应包头

# ======================== 响应消息类型 ========================
RESP_SUCCESS = 0x01
RESP_ERROR = 0x02
RESP_WARNING = 0x03
RESP_PROCESSING = 0x04

# ======================== 接收端实现 ========================
class CommandImageReceiver:
    def __init__(self):
        self.command_buffer = queue.Queue(maxsize=10)
        self.image_buffer = queue.Queue(maxsize=10)
        self.cond = threading.Condition()
        self.running = True
        self.receiver_thread = threading.Thread(target=self._receive_thread, daemon=True)
        self.receiver_thread.start()
        self.client_socket = None
        self.current_addr = None
    
    def _receive_thread(self):
        """后台接收线程"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('0.0.0.0', PORT))
        server_socket.listen(1)
        logger.info(f"接收端已启动，监听端口 {PORT}...")
        
        try:
            while self.running:
                self.client_socket, self.current_addr = server_socket.accept()
                logger.info(f"接受来自 {self.current_addr} 的连接")
                
                try:
                    while self.running:
                        # 读取包头 (3字节)
                        header = self._recv_exactly(self.client_socket, 3)
                        if not header:
                            break
                            
                        # 读取数据长度 (4字节)
                        data_len_bytes = self._recv_exactly(self.client_socket, 4)
                        if not data_len_bytes:
                            break
                        data_len = struct.unpack('>I', data_len_bytes)[0]
                        
                        # 读取完整数据
                        data = self._recv_exactly(self.client_socket, data_len)
                        if not data:
                            break
                        
                        # 处理指令+图像数据
                        if header == CMD_HEADER:
                            success = self._process_command_image(data)
                            # self._send_response(success, "指令+图像处理完成" if success else "指令+图像处理失败")
                        
                        # 处理仅图像数据
                        elif header == IMG_HEADER:
                            success = self._process_image(data)
                            # self._send_response(success, "图像处理完成" if success else "图像处理失败")
                        
                        else:
                            logger.warning(f"未知包头: {header.hex()}")
                            # self._send_response(False, f"未知包头: {header.hex()}")
                except (ConnectionResetError, BrokenPipeError) as e:
                    logger.error(f"与客户端 {self.current_addr} 的连接中断: {str(e)}")
                finally:
                    self.client_socket.close()
                    self.client_socket = None
                    logger.info(f"与客户端 {self.current_addr} 的连接已关闭")
        finally:
            server_socket.close()
    
    def _process_command_image(self, data):
        """处理指令+图像数据包"""
        try:
            # 解压缩并反序列化
            decompressed = zlib.decompress(data)
            command, images = pickle.loads(decompressed)
            
            # 验证数据类型
            if not isinstance(command, str):
                raise ValueError("命令部分不是字符串")
            if not isinstance(images, list):
                raise ValueError("图像部分不是列表")
            
            logger.info(f"收到指令+图像: 指令='{command}', 图像数量={len(images)}")
            
            # 将指令和图像放入缓冲区
            with self.cond:
                self.command_buffer.put(command)
                self.image_buffer.put(images)
                self.cond.notify()  # 通知主线程
                
            return True
        except Exception as e:
            logger.error(f"处理指令+图像失败: {str(e)}")
            return False
    
    def _process_image(self, data):
        """处理仅图像数据包"""
        try:
            # 解压缩并反序列化
            decompressed = zlib.decompress(data)
            images = pickle.loads(decompressed)
            
            if not isinstance(images, list):
                raise ValueError("图像部分不是列表")
            
            logger.info(f"收到图像序列, 数量={len(images)}")
            
            # 将图像放入缓冲区
            with self.cond:
                self.image_buffer.put(images)
                self.cond.notify()  # 通知主线程
                
            return True
        except Exception as e:
            logger.error(f"处理图像失败: {str(e)}")
            return False
    
    def _send_response(self, success, message):
        """发送响应给客户端"""
        if not self.client_socket:
            logger.warning("无法发送响应: 没有活动的客户端连接")
            return
        
        try:
            # 创建响应数据包
            status = RESP_SUCCESS if success else RESP_ERROR
            response_data = {
                "status": status,
                "timestamp": time.time(),
                "message": message
            }
            
            # 序列化并压缩响应
            serialized = pickle.dumps(response_data)
            compressed = zlib.compress(serialized)
            
            # 构造响应包: 包头(3B) + 数据长度(4B) + 压缩数据
            data_len = struct.pack('>I', len(compressed))
            packet = RESP_HEADER + data_len + compressed
            
            # 发送响应
            self.client_socket.sendall(packet)
            logger.info(f"已发送响应: {message}")
        except Exception as e:
            logger.error(f"发送响应失败: {str(e)}")
    
    def _recv_exactly(self, sock, length):
        """从socket接收指定长度的数据"""
        data = b''
        while len(data) < length:
            chunk = sock.recv(length - len(data))
            if not chunk:
                return None
            data += chunk
        return data
    
    # def get_next_command_image(self, timeout=10):
    def get_next_command_image(self):
        """获取下一个指令和图像组合"""
        # print("wait 之前")
        with self.cond:
            ret = self.cond.wait()
            # print("wait 之后")
            
            print("队列长度：",self.command_buffer.qsize(),self.image_buffer.qsize())
            
            if(self.command_buffer.qsize()>0):
                command = self.command_buffer.get()
                images = self.image_buffer.get()
                return images,command
            else:
                images = self.image_buffer.get()
                return images,None
    
    def stop(self):
        """停止接收器"""
        self.running = False
        with self.cond:
            self.cond.notify_all()

def receiver_main():
    """接收端主程序"""
    receiver = CommandImageReceiver()
    logger.info("接收端已启动，等待数据...")
    
    try:
        while True:
            # 等待并获取下一个指令+图像组合
            command, images = receiver.get_next_command_image(timeout=30)
            
            if command is None and images is None:
                logger.info("等待超时，继续等待...")
                continue
                
            if command:
                logger.info(f"\n=== 处理指令 ===")
                logger.info(f"指令内容: '{command}'")
                
                # 根据指令执行不同操作
                if command == "START_PROCESSING":
                    logger.info("启动图像处理流程...")
                elif command == "SAVE_IMAGES":
                    logger.info("保存接收到的图像...")
                else:
                    logger.warning(f"未知指令: '{command}'")
            
            if images:
                logger.info(f"\n=== 处理图像序列 ===")
                logger.info(f"收到 {len(images)} 张图像")
                
                # 处理每张图像
                timestamp = int(time.time())
                for i, img in enumerate(images):
                    if isinstance(img, np.ndarray):
                        logger.info(f"图像 {i+1}: 尺寸 {img.shape}, 数据类型 {img.dtype}")
                        
                        # 示例: 保存第一张图像
                        if i == 0 and command == "SAVE_IMAGES":
                            filename = f"received_img_{timestamp}.png"
                            cv2.imwrite(filename, img)
                            logger.info(f"已保存图像: {filename}")
                    else:
                        logger.warning(f"图像 {i+1} 不是numpy数组")
                
                logger.info("图像处理完成\n")
                
            time.sleep(1)
            
            receiver._send_response(True, "图像处理完成")
                
    except KeyboardInterrupt:
        logger.info("\n接收端正在停止...")
    finally:
        receiver.stop()


# if __name__ == "__main__":
#     import sys
    
#     # 设置日志级别
#     logger.setLevel(logging.INFO)
#     receiver_main()
