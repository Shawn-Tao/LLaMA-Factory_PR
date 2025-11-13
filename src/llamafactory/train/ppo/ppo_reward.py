import regex as re
import numpy as np    

class RewardManager:
        
    # 一般而言，completion 代表模型生成的内容，solution 代表正确答案或参考答案。
    def direction_estimate_reward(self, completions, solution, priveleged_info_batch):
        reward = 0
        print("completions:",completions)
        print("solution:",solution)
        print("priveleged_info_batch:",priveleged_info_batch)
        
        # # completion_contents = [completion[0]["content"] for completion in completions]
        # for content, sol in zip(completions, solution):
        #     print("*********content",content)
        #     print("*********sol",sol)
        #     exit()
        
        # 匹配方向，如果运动方向正确，给1.0，如果运动方向错误，根据错误的情况来分配奖励
        if re.search(r'move forward', solution):
            if re.search(r'move forward', completions):
                reward =  1.0
            else:
                # 检查转向角度，
                turn_left_match = re.search(r'turn left|turn right', completions)
                if turn_left_match:
                    degree_match = re.search(r'degrees', completions)
                    degree_str = completions[turn_left_match.end()+1:degree_match.start()-1]
                    degree = float(degree_str)
                    if degree == 15:
                        reward =  0.5
                    elif degree == 30:
                        reward =  0.3
                    else:
                        reward =  0.0
        elif re.search(r'turn left', solution):
            if re.search(r'turn left', completions):
                reward = 1.0
            elif re.search(r'move forward', completions):
                reward =  0.5
            elif re.search(r'turn right', completions):
                reward =  0.0
        elif re.search(r'turn right', solution):
            if re.search(r'turn right', completions):
                reward =  1.0
            elif re.search(r'move forward', completions):
                reward= 0.5
            elif re.search(r'turn left', completions):
                reward= 0.0
                
        return reward


    def navigation_format_reward(self,completions, solution, priveleged_info_batch):
        reward = 0.0
        # completion_contents = [completion[0]["content"] for completion in completions]
        # completion_contents = completions[0][0]["content"]
        # print("completion_contents:",completion_contents)
        pattern = r'^(?:move forward (?:\.\d+|\d+(?:\.\d*)?) meters|turn left \d+ degrees|turn right \d+ degrees|stop)$'
        if(re.match(pattern, completions)):
            reward =  1.0
        else:
            reward =  -1.0
        
        return reward

        
    # !首先根据航向、位置、动作推理下一步是原理目标还是接=近目标，然后来计算reward
    # !要求保存图像和指令之外，还要保存目标位置、航向、当前位置等信息，尝试看看VLN-CE能给我们提供什么东西
    # ! 现在考虑只计算一个远离的惩罚。
    def aim_distance_reward(self, completions, solution, priveleged_info_batch):
        agent_position = priveleged_info_batch["agent_position"][0]
        goal_position = priveleged_info_batch["goal_position"][0]
        agent_heading = priveleged_info_batch["agent_heading"][0]
        distance_to_goal =  priveleged_info_batch["distance_to_goal"][0]
        
        # 坐标系方向为，z轴朝下，x轴朝右，agent_heading 是用弧度数值表示的朝向。，以z轴负方向为0度，逆时针为正方向
        # 将agent_heading修改为 zox坐标系下的标准角度表示
        
        # ! 有两种表示方法，正的和负的，分别是 0 到 180 和 0 到 -180，这里先把负的转正的，方便后面处理
        if(agent_heading < 0):
            agent_heading = agent_heading + 2 * np.pi
        else:
            agent_heading = agent_heading
            
        # ! 先减少180度，将正方向由z轴负方向改为z轴正方向，然后对2pi取模，
        agent_heading = (agent_heading - np.pi) % (2 * np.pi)  # 将角度转换为0-2pi范围内的值
        
        # 计算旋转矩阵
        rotation_matrix_2d = np.array([
            [np.cos(agent_heading), -np.sin(agent_heading)],
            [np.sin(agent_heading), np.cos(agent_heading)]
        ])
        
        goal_position_in_zox = [goal_position[2],goal_position[0]]  # 只取z和x坐标
        
        # completion_contents = completions[0][0]["content"]
        
        # for content, sol in zip(completion_contents, solution):
        #     # 匹配方向，如果运动方向正确，给1.0，如果运动方向错误，根据错误的情况来分配奖励
        #     if re.search(r'move forward', sol):
        #         if re.search(r'move forward', content):
        #             reward =  1.0
        
        reward = 0.0
        if re.match(completions, r'move forward \d+ meter'):
            # next_position = agent_position + np.dot(rotation_matrix_2d, np.array([1, 0])) * float(completions.split()[2])
            next_position = agent_position + np.dot(rotation_matrix_2d, np.array([1, 0])) * float(completions.split()[2])
            next_distance_to_goal = np.linalg.norm(next_position - goal_position)
            
            # 距离相减，如果接近了目标，就给正奖励，远离目标就给负奖励
            reward = (distance_to_goal - next_distance_to_goal) * 3.0
            
            # # 如果下一步移动后距离目标更远，给负奖励
            # if next_distance_to_goal > distance_to_goal:
            # #     reward = 1.0
            # # else:
            #     reward = -1.0
        
        return reward


    # 核心奖励函数 1 --> 增强到达判断，到达后停止
    def stop_in_domain_reward(self, completions, solution, priveleged_info_batch):
        distance_to_goal =  priveleged_info_batch["distance_to_goal"][0]
        reward = 0.0
        if(completions == "stop"):
            #越接近目标，奖励越大，奖励最大为 5.0，从1 开始 到0 分别是 0，5.0
            reward = 5.0 - distance_to_goal * 5.0
        else:
            if(distance_to_goal > 1):
                reward = 0.0
            else:
                # 如果距离小于1米，判断不停止，就给一个惩罚，惩罚和距离有关，和上述停止奖励对称
                reward = distance_to_goal * 5.0 - 5.0
                
        return reward


    def test_rewards(self, completions, solution, priveleged_info_batch):
        # print(completions)
        # print(solution)
        # print(priveleged_info_batch)
        # exit()
        return 0.0