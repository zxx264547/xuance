import numpy as np
import pandas as pd
from gym.spaces import Box
from torch.utils.tensorboard import SummaryWriter
from xuance.environment import RawMultiAgentEnv
import pandapower as pp
from scipy.io import loadmat
from datetime import datetime
import os



class IEEE13(RawMultiAgentEnv):
    def __init__(self, env_config, v0=1, vmax=1.05, vmin=0.95):
        super(IEEE13, self).__init__()
        self.obs_dim = 5
        self.action_dim = 2
        # 初始化电网
        self.network = create_123bus()
        self.injection_bus = np.array([10, 11, 16, 20, 33, 36, 48, 59, 66, 75, 83, 92, 104, 61]) - 1

        self.env_id = env_config.env_id
        self.num_agents = len(self.injection_bus)
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.state_space = Box(-np.inf, np.inf, shape=[self.obs_dim, ])
        self.observation_space = {agent: Box(-np.inf, np.inf, shape=[self.obs_dim, ]) for agent in self.agents}
        self.action_space = {agent: Box(-np.inf, np.inf, shape=[self.action_dim, ]) for agent in self.agents}
        self._current_step = 0
        # 将节点和智能体一一对应
        self.agent_to_bus = dict(zip(self.agents, self.injection_bus))
        self.v0 = v0
        self.vmax = vmax
        self.vmin = vmin
        self.load0_p = np.copy(self.network.load['p_mw'])
        self.load0_q = np.copy(self.network.load['q_mvar'])
        self.gen0_p = np.copy(self.network.sgen['p_mw'])
        self.gen0_q = np.copy(self.network.sgen['q_mvar'])
        self.state = {}
        self.load_p = pd.read_csv('F:/xuance\myCode\data\load_p.csv')
        self.load_q = pd.read_csv('F:/xuance\myCode\data\load_q.csv')
        self.pv_p = pd.read_csv('F:/xuance\myCode\data\PV_p')

        self.max_episode_steps = self.load_p.size - 1


        # 初始化 TensorBoard 的 SummaryWriter
        # 获取当前时间并格式化为字符串
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        # 在 "logs/IEEE13" 下创建一个带时间戳的子文件夹
        base_log_dir = "logs/IEEE13"
        log_dir = os.path.join(base_log_dir, current_time)
        self.writer = SummaryWriter(log_dir)

    def get_env_info(self):
        return {'state_space': self.state_space,
                'observation_space': self.observation_space,
                'action_space': self.action_space,
                'agents': self.agents,
                'num_agents': self.num_agents,
                'max_episode_steps': self.max_episode_steps}

    def avail_actions(self):
        return None

    def agent_mask(self):
        """Returns boolean mask variables indicating which agents are currently alive."""
        return {agent: True for agent in self.agents}

    def state(self):
        """Returns the global state of the environment."""
        return self.state_space.sample()

    def reset(self):
        # observation = {agent: self.observation_space[agent].sample() for agent in self.agents}
        info = {}

        self._current_step = 0

        self.network.sgen['p_mw'] = 0.0
        self.network.sgen['q_mvar'] = 0.0
        self.network.load['p_mw'] = 0.0
        self.network.load['q_mvar'] = 0.0

        # self.network.sgen.at[0, 'p_mw'] = -0.8 * np.random.uniform(15, 60)
        self.network.sgen.at[1, 'p_mw'] = -0.8 * np.random.uniform(10, 45)
        self.network.sgen.at[2, 'p_mw'] = -0.8 * np.random.uniform(10, 55)
        self.network.sgen.at[3, 'p_mw'] = -0.8 * np.random.uniform(10, 30)
        self.network.sgen.at[4, 'p_mw'] = -0.6 * np.random.uniform(1, 35)
        self.network.sgen.at[5, 'p_mw'] = -0.5 * np.random.uniform(2, 25)
        self.network.sgen.at[6, 'p_mw'] = -0.8 * np.random.uniform(2, 30)
        self.network.sgen.at[7, 'p_mw'] = -0.9 * np.random.uniform(1, 10)
        self.network.sgen.at[8, 'p_mw'] = -0.7 * np.random.uniform(1, 15)
        self.network.sgen.at[9, 'p_mw'] = -0.5 * np.random.uniform(1, 30)
        self.network.sgen.at[10, 'p_mw'] = -0.3 * np.random.uniform(1, 20)
        self.network.sgen.at[11, 'p_mw'] = -0.5 * np.random.uniform(1, 20)
        self.network.sgen.at[12, 'p_mw'] = -0.4 * np.random.uniform(1, 20)
        self.network.sgen.at[13, 'p_mw'] = -0.4 * np.random.uniform(2, 10)
        # 进行潮流计算
        pp.runpp(self.network, algorithm='bfsw', init='dc')

        # 获取状态信息
        for bus_idx in self.network.bus.index:
            # 提取发电机有功和无功功率
            gen_p_mw = self.network.res_gen[self.network.gen['bus'] == bus_idx]['p_mw'].sum()
            gen_q_mvar = self.network.res_gen[self.network.gen['bus'] == bus_idx]['q_mvar'].sum()
            # 提取节点电压
            voltage_pu = self.network.res_bus.loc[bus_idx, 'vm_pu']
            # 提取负载有功和无功功率
            load_p_mw = self.network.res_load[self.network.load['bus'] == bus_idx]['p_mw'].sum()
            load_q_mvar = self.network.res_load[self.network.load['bus'] == bus_idx]['q_mvar'].sum()
            # 将数据存储为一个数组
            node_data = np.array([gen_p_mw, gen_q_mvar, voltage_pu, load_p_mw, load_q_mvar])
            # 将数组添加到 self.state 字典中
            self.state[bus_idx] = node_data

        # 获得每个智能体的观测值
        observation = {}
        for agent in self.agents:
            observation[agent] = self.state[self.agent_to_bus[agent]]
        return observation, info

    def step(self, action_dict):
        self._current_step += 1

        # 应用智能体的动作
        for i, agent in enumerate(self.agents):
            # 从 action_dict 中获取该智能体的动作
            action = action_dict[agent]
            # 将动作值赋给 `self.network.sgen` 的 q_mvar 列
            if action is not None:
                # 将动作值赋给 self.network.sgen 的 q_mvar 列
                self.network.sgen.at[i, 'q_mvar'] = action[0]
                # self.network.sgen.at[i, 'p_mw'] = self.pv_p['0'][self._current_step] / 50
                self.network.sgen.at[i, 'p_mvar'] = action[1]
            else:
                print(f"No action found for {agent}")

        # 设置每个节点的负载功率
        for index in self.network.load.index:
            self.network.load.at[index, 'p_mw'] = self.load_p['0'][self._current_step] / 50
            self.network.load.at[index, 'q_mvar'] = self.load_q['0'][self._current_step] / 50
        # self.writer.add_scalar(f"load_p", self.load_p['p'][self._current_step, 0] / 50, self._current_step)
        # self.writer.add_scalar(f"load_q", self.load_q['q'][self._current_step, 0] / 50, self._current_step)
        # self.writer.add_scalar(f"pv_p", self.pv_p['0'][self._current_step] / 50, self._current_step)

        # 进行潮流计算
        pp.runpp(self.network, algorithm='bfsw', init='dc')

        # 获取状态信息
        for bus_idx in self.network.bus.index:
            # 提取发电机有功和无功功率
            gen_p_mw = self.network.res_gen[self.network.gen['bus'] == bus_idx]['p_mw'].sum()
            gen_q_mvar = self.network.res_gen[self.network.gen['bus'] == bus_idx]['q_mvar'].sum()
            # 提取节点电压
            voltage_pu = self.network.res_bus.loc[bus_idx, 'vm_pu']
            # 提取负载有功和无功功率
            load_p_mw = self.network.res_load[self.network.load['bus'] == bus_idx]['p_mw'].sum()
            load_q_mvar = self.network.res_load[self.network.load['bus'] == bus_idx]['q_mvar'].sum()
            # 将数据存储为一个数组
            node_data = np.array([gen_p_mw, gen_q_mvar, voltage_pu, load_p_mw, load_q_mvar])
            # 将数组添加到 self.state 字典中
            self.state[bus_idx] = node_data

        # 获得每个智能体的观测值
        observation = {}
        for agent in self.agents:
            observation[agent] = self.state[self.agent_to_bus[agent]]

        # 计算rewards = 智能体自己的电压越限 + 所有节点的电压越限 + 总网损
        total_violation = 0
        power_losses = 0
        rewards = {}
        for agent in self.agents:
            # 智能体自己的电压越限
            voltage = observation[agent][2]
            low_voltage_violation = max(self.vmin - voltage, 0)
            high_voltage_violation = max(voltage - self.vmax, 0)
            voltage_violation_self = low_voltage_violation + high_voltage_violation
            voltage_penalty = -voltage_violation_self * 1000
            # 所有节点的电压越限
            total_violation = np.sum(np.maximum(self.network.res_bus.vm_pu.values - self.vmax, 0)
                                      + np.maximum(self.vmin - self.network.res_bus.vm_pu.values, 0))
            # 总网损
            power_losses = self.network.res_line.pl_mw + self.network.res_line.ql_mvar
            loss_penalty = -np.sum(power_losses)

            # 动作惩罚
            # Action cost (encourages small actions)
            # action_cost = -np.abs(action_dict[agent]) * 10

            rewards[agent] = np.array(voltage_penalty + loss_penalty - total_violation).item()

        # 记录电压越限率和网损
        self.writer.add_scalar(f"total_violation", total_violation, self._current_step)
        self.writer.add_scalar(f"power_losses", power_losses.sum(), self._current_step)

        # 初始化每个智能体的 terminated 状态为 False
        terminated = {agent: False for agent in self.agents}
        # # 遍历每个智能体的电压状态，检查是否在指定的范围内
        # for agent in self.agents:
        #     # 判断当前智能体的电压是否在 0.9499 到 1.0501 之间
        #     if 0.9499 <= observation[agent][2] <= 1.0501:
        #         terminated[agent] = True
        truncated = False if self._current_step < self.max_episode_steps else True

        # 记录每个节点的电压信息和每个智能体的动作到 TensorBoard
        for agent_id, action in action_dict.items():
            for i, action_value in enumerate(action):
                self.writer.add_scalar(f"Agent_{agent_id}/Action_{i}", action_value, self._current_step)

        # 自定义信息
        info = {
            "agent_actions": action_dict,
            "current_step": self._current_step
        }

        return observation, rewards, terminated, truncated, info

    def render(self, *args, **kwargs):
        return np.ones([64, 64, 64])

    def close(self):
        # 关闭 TensorBoard 记录器
        self.writer.close()


def create_123bus():
    pp_net = pp.converter.from_mpc('F:/xuance/myCode/pandapower models/case_123.mat', casename_mpc_file='case_mpc')

    pp_net.sgen['p_mw'] = 0.0
    pp_net.sgen['q_mvar'] = 0.0

    injection_bus = np.array([10, 11, 16, 20, 33, 36, 48, 59, 66, 75, 83, 92, 104, 61]) - 1
    for bus in injection_bus:
        pp.create_sgen(pp_net, bus, p_mw=1, q_mvar=0)

    return pp_net
