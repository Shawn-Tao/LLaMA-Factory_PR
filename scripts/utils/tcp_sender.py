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
MAX_IMAGES = 5       # 最大图像数量
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

# ======================== 发送端实现 ========================
class CommandImageSender:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.client_socket = None
        self.pending_command = None  # 待发送的指令
        self.response_queue = queue.Queue(maxsize=5)
        self.response_thread = None
        self.running = False
        self.response_condition = threading.Condition()
    
    def connect(self):
        """连接到接收端"""
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # self.client_socket.settimeout(10.0)  # 设置连接超时
            self.client_socket.connect((self.host, self.port))
            
            # 启动响应接收线程
            self.running = True
            self.response_thread = threading.Thread(target=self._response_listener, daemon=True)
            self.response_thread.start()
            
            logger.info(f"已连接到接收端 {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"连接失败: {str(e)}")
            return False
    
    def disconnect(self):
        """断开连接"""
        self.running = False
        if self.client_socket:
            self.client_socket.close()
            self.client_socket = None
            logger.info("已断开连接")
        
        if self.response_thread and self.response_thread.is_alive():
            self.response_thread.join(timeout=1.0)
    
    def set_command(self, command):
        """设置待发送的指令（不立即发送）"""
        if not isinstance(command, str):
            raise ValueError("指令必须是字符串")
        self.pending_command = command
        logger.info(f"指令已设置: '{command}'")
    
    def _response_listener(self):
        """监听来自接收端的响应"""
        while self.running and self.client_socket:
            try:
                # 读取包头 (3字节)
                header = self._recv_exactly(3)
                if not header:
                    break
                
                # 检查是否为响应包
                if header != RESP_HEADER:
                    logger.warning(f"收到非响应包: {header.hex()}")
                    continue
                
                # 读取数据长度 (4字节)
                data_len_bytes = self._recv_exactly(4)
                if not data_len_bytes:
                    break
                data_len = struct.unpack('>I', data_len_bytes)[0]
                
                # 读取完整数据
                data = self._recv_exactly(data_len)
                if not data:
                    break
                
                # 解压缩并反序列化响应
                try:
                    decompressed = zlib.decompress(data)
                    response = pickle.loads(decompressed)
                    with self.response_condition:
                        self.response_queue.put(response)
                        self.response_condition.notify()  # 通知主线程
                        logger.info(f"收到响应: {response.get('message', '无消息')}")
                except Exception as e:
                    logger.error(f"处理响应失败: {str(e)}")
                    
            except socket.timeout:
                # 超时是正常的，继续监听
                continue
            except (ConnectionResetError, BrokenPipeError) as e:
                logger.error(f"连接中断: {str(e)}")
                break
            except Exception as e:
                logger.error(f"响应监听错误: {str(e)}")
                break
    
    def _recv_exactly(self, length):
        # """从socket接收指定长度的数据"""
        # data = b''
        # start_time = time.time()
        # while len(data) < length:
        #     try:
        #         # 设置剩余超时时间
        #         remaining = RESPONSE_TIMEOUT - (time.time() - start_time)
        #         if remaining <= 0:
        #             raise socket.timeout()
                
        #         self.client_socket.settimeout(remaining)
        #         chunk = self.client_socket.recv(length - len(data))
        #         if not chunk:
        #             return None
        #         data += chunk
        #     except socket.timeout:
        #         logger.warning(f"接收数据超时 ({len(data)}/{length} 字节)")
        #         return None
        # return data
        
        """从socket接收指定长度的数据"""
        data = b''
        while len(data) < length:
            chunk = self.client_socket.recv(length - len(data))
            if not chunk:
                return None
            data += chunk
        return data
    
    def get_response(self, timeout=RESPONSE_TIMEOUT):
        """获取响应"""
        try:
            return self.response_queue.get(timeout=timeout)
        except queue.Empty:
            logger.warning("获取响应超时")
            return None
        
    def get_aciton_str(self):
        with self.response_condition:
            self.response_condition.wait()
            action_str = self.response_queue.get()
        return action_str
    
    def send_images(self, images, include_command=True):
        """发送图像序列，可选择包含指令"""
        if not self.client_socket:
            logger.error("未连接接收端")
            return False
        
        # 验证并预处理图像
        valid_images = []
        for i, img in enumerate(images):
            if not isinstance(img, np.ndarray):
                logger.warning(f"图像 {i} 不是numpy数组，已跳过")
                continue
            
            # 调整图像尺寸
            if img.shape[:2] != IMAGE_SIZE:
                logger.warning(f"图像 {i} 尺寸 {img.shape} 不匹配，调整为 {IMAGE_SIZE}")
                img = cv2.resize(img, IMAGE_SIZE)
            
            # 确保是单通道或三通道
            if len(img.shape) == 2:  # 单通道
                img = np.expand_dims(img, axis=-1)  # 添加通道维度
            elif img.shape[2] > 3:  # 多通道，取前三个
                img = img[:, :, :3]
            
            valid_images.append(img)
            
            if len(valid_images) >= MAX_IMAGES:
                logger.warning(f"达到最大图像数量限制 ({MAX_IMAGES})")
                break
        
        if not valid_images:
            logger.error("无有效图像可发送")
            return False
        
        try:
            # 序列化数据
            if include_command and self.pending_command:
                # 发送指令+图像
                data_to_send = (self.pending_command, valid_images)
                self.pending_command = None  # 清除待发送指令
                serialized = pickle.dumps(data_to_send)
                header = CMD_HEADER
            else:
                # 仅发送图像
                data_to_send = valid_images
                serialized = pickle.dumps(data_to_send)
                header = IMG_HEADER
            
            # 压缩数据
            compressed = zlib.compress(serialized)
            
            # 构造数据包: 包头(3B) + 数据长度(4B) + 压缩数据
            data_len = struct.pack('>I', len(compressed))
            packet = header + data_len + compressed
            
            # 发送数据
            self.client_socket.sendall(packet)
            
            img_count = len(valid_images)
            if include_command and header == CMD_HEADER:
                logger.info(f"已发送指令+图像: {img_count}张图像")
            else:
                logger.info(f"已发送图像: {img_count}张图像")
                
            return True
        except Exception as e:
            logger.error(f"发送失败: {str(e)}")
            return False

def sender_main():
    """发送端主程序"""
    sender = CommandImageSender(HOST, PORT)
    if not sender.connect():
        return
    
    try:
        # 设置指令（但不立即发送）
        sender.set_command("START_PROCESSING")
        
        # 创建测试图像1
        images1 = [
            np.random.randint(0, 256, (*IMAGE_SIZE, 3), dtype=np.uint8),  # 随机彩色图像
            np.zeros((*IMAGE_SIZE, 3), dtype=np.uint8),                   # 黑色图像
        ]
        
        # 第一次发送：包含指令和图像
        if sender.send_images(images1, include_command=True):
            pass
            # # 等待并处理响应
            # response = sender.get_response()
            # if response:
            #     status = response.get("status", RESP_ERROR)
            #     message = response.get("message", "无消息")
            #     logger.info(f"收到响应: 状态={status}, 消息='{message}'")
        
        # 等待片刻
        time.sleep(1)
        
        # 设置新指令（但这次不会随图像发送）
        sender.set_command("SAVE_IMAGES")
        
        # 创建测试图像2
        images2 = [np.full((*IMAGE_SIZE, 3), 255, dtype=np.uint8)]
        
        images2.append(np.zeros((*IMAGE_SIZE, 3), dtype=np.uint8))
        
        # 第二次发送：仅发送图像
        if sender.send_images(images2, include_command=False):
            pass
            # response = sender.get_response()
            # if response:
            #     status = response.get("status", RESP_ERROR)
            #     message = response.get("message", "无消息")
            #     logger.info(f"收到响应: 状态={status}, 消息='{message}'")
        
        # 等待片刻
        time.sleep(1)
        
        # 第三次发送：包含新指令和图像
        if sender.send_images(images1, include_command=True):
            pass
            # response = sender.get_response()
            # if response:
            #     status = response.get("status", RESP_ERROR)
            #     message = response.get("message", "无消息")
            #     logger.info(f"收到响应: 状态={status}, 消息='{message}'")
        
        logger.info("所有数据已发送")
        
        
        time.sleep(10)
        
    except KeyboardInterrupt:
        logger.info("发送被用户中断")
    finally:
        sender.disconnect()

if __name__ == "__main__":
    import sys
    
    # 设置日志级别
    logger.setLevel(logging.INFO)
    sender_main()
