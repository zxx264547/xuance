import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 16})


def scale_data(data, a, b):
    """
    将输入数据放缩到指定的区间 [a, b] 之间。

    :param data: 输入数据，为一个 NumPy 数组或列表。
    :param a: 放缩后的最小值。
    :param b: 放缩后的最大值。
    :return: 返回放缩后的数据。
    """
    data = np.array(data)  # 确保数据为 NumPy 数组类型
    min_val = np.min(data)
    max_val = np.max(data)

    # 处理边界情况，避免除以零
    if max_val == min_val:
        return np.full(data.shape, (a + b) / 2)  # 当所有数据相同时，返回一个常数数组

    # 放缩数据到 [a, b] 之间
    scaled_data = (b - a) * (data - min_val) / (max_val - min_val) + a

    return scaled_data

def create_sigmod(a, b, x_range=(0, 1000)):
    """
    生成一个类似于上传图像形状的函数，曲线为S型，经过原点，并在 x 增大时稳定到一个值 b。

    :param a: 对称轴的位置，表示曲线的上升速率变化的中间位置。
    :param b: 函数在 x 增大时趋于的稳定值。
    :param x_range: x 的取值范围，默认从 0 到 1000。
    :return: 返回 y 的值（函数值），为一个 1000 维的数组。
    """
    x = np.linspace(x_range[0], x_range[1], 1000)

    # 使用平滑函数模拟S型曲线，经过原点并趋于b
    k = 1 / a  # 控制曲线的平缓程度
    y = b * (1 - np.exp(-k * x)) / (1 + np.exp(-k * (x - a)))  # 平滑曲线，经过原点并趋于 b

    plt.figure()
    plt.plot(y)
    plt.show()
    return y
def create_data(a, max_value, c, x_range=(0, 1000)):
    """
    生成一个类似于上传图像形状的函数，曲线具有单个峰值，经过原点，并在 x 增大时稳定到一个值 c。

    :param a: 对称轴的位置，表示曲线的峰值位置。
    :param max_value: 该函数的最大值。
    :param c: 函数在 x 增大时趋于的稳定值。
    :param x_range: x 的取值范围，默认从 0 到 1000。
    :return: 返回 y 的值（函数值），为一个 1000 维的数组。
    """
    x = np.linspace(x_range[0], x_range[1], 1000)

    # 定义一个函数，模拟图中的形状，经过原点，并在 a 处有峰值
    scale = a / 5  # 控制宽度，值可以根据需要调整
    y_peak = (x / scale) ** 2 * np.exp(-x / scale)  # gamma-like function，表示峰值部分

    # 归一化处理，使曲线达到设定的最大值 max_value
    y_peak = y_peak / np.max(y_peak) * (max_value - c)

    # 添加稳定值 c，使得 y 在 x 增大时逐渐趋于 c
    y = y_peak + c

    return y

def process_data(original_data, slope_factor, final_value_offset):
    """
    Process the input data by modifying the decay rate (slope) and the final stable value.

    Parameters:
    - original_data (numpy array): The original time-series data.
    - slope_factor (float): Multiplier to adjust the decay rate of the data.
    - final_value_offset (float): Value to subtract from the final stable value.

    Returns:
    - numpy array: Processed data with adjusted decay rate and stable value.
    """
    # Ensure the data is a numpy array
    original_data = np.array(original_data)

    # Create a time array based on the length of the data
    steps = np.arange(len(original_data))

    # Apply the slope modification (faster decay)
    processed_data = original_data * (np.exp(-slope_factor * steps) + 1)

    # Adjust the final stable value
    processed_data -= final_value_offset

    processed_data = processed_data + np.abs(processed_data.min())

    return processed_data
flag = 'reward'

# VVR_MADDPG = pd.read_csv('loss_ad_maddpg.csv')
# VVR_MASAC = pd.read_csv('loss_masac.csv')
# PL_1 = pd.read_csv('PL_1.csv')
# PL_MADDPG = pd.read_csv('PL_MADDPG.csv')
#
# # 绘制图形
# plt.figure(figsize=(8, 6))
# plt.plot(VVR_MADDPG['Step'], VVR_MADDPG['Value'], label='VVR_MADDPG')
# plt.plot(VVR_MASAC['Step'], VVR_MASAC['Value'], label='VVR_MASAC')
# # 设置图表标题和标签
# plt.title('Step vs Value')
# plt.xlabel('Step')
# plt.ylabel('Value')
#
# # 绘制图形
# plt.figure(figsize=(8, 6))
# plt.plot(PL_1['Wall time'], PL_1['Value'], label='PL_1')
# # 设置图表标题和标签
# plt.title('Step vs Value')
# plt.xlabel('Step')
# plt.ylabel('Value')

'''
智能体的动作输出
'''
if flag == 'action':
    # 定义智能体的 CSV 文件路径
    csv_files = {
        '光伏1': 'pv1_q.csv',
        '光伏2': 'pv2_q.csv',
        '光伏3': 'pv3_q.csv',
        '光伏4': 'pv4_q.csv',
        '光伏5': 'pv5_q.csv',
        '光伏6': 'pv6_q.csv',
        '光伏7': 'pv7_q.csv',
        '光伏8': 'pv8_q.csv',
        '光伏9': 'pv9_q.csv',
        '光伏10': 'pv10_q.csv',
        '光伏11': 'pv11_q.csv',
        '光伏12': 'pv12_q.csv',
        '光伏13': 'pv13_q.csv',
        '储能1': 'es1_q.csv',
        '储能2': 'es2_q.csv',
        '储能3': 'es3_q.csv',
    }

    # 读取每个文件的 "Value" 列，并取前 8640 行
    agents_data = {}
    for agent_name, file_path in csv_files.items():
        data = pd.read_csv(file_path)
        # 提取 "Value" 列的前 8640 行
        agents_data[agent_name] = data['Value'][:262].values * 0.8

    step = np.linspace(1, 8640, 262)

    # 创建子图
    num_agents = len(agents_data)
    cols = 4  # 每行显示4个子图
    rows = (num_agents + cols - 1) // cols  # 根据智能体数量动态计算行数

    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 3))
    axes = axes.flatten()  # 将轴展平成一维列表，便于索引

    # 绘制每个智能体的动作曲线
    for i, (agent_name, action_data) in enumerate(agents_data.items()):
        axes[i].plot(step, action_data, label="无功输出")
        axes[i].set_title(agent_name)
        axes[i].set_xlabel('Step')
        axes[i].set_ylabel('功率(p.u.)')
        axes[i].grid(True, linestyle='--', alpha=0.7)
        axes[i].legend(loc='upper left')

    es1_p = pd.read_csv('es1_p.csv')
    action_data = es1_p['Value'][:262].values
    axes[13].plot(step, action_data, label="有功输出")
    axes[13].legend(loc='upper left')

    es2_p = pd.read_csv('es2_p.csv')
    action_data = es1_p['Value'][:262].values
    axes[14].plot(step, action_data, label="有功输出")
    axes[14].legend(loc='upper left')


    es3_p = pd.read_csv('es3_p.csv')
    action_data = es1_p['Value'][:262].values
    axes[15].plot(step, action_data, label="有功输出")
    axes[15].legend(loc='upper left')


    # 隐藏多余的子图
    for j in range(len(agents_data), len(axes)):
        fig.delaxes(axes[j])


    # 调整布局并显示
    plt.tight_layout()
    plt.show()

elif flag == 'violation':
    step = np.linspace(1, 10000, 161)

    violation_1 = pd.read_csv('violation_1.csv')
    violation_2 = pd.read_csv('violation_2.csv')
    violation_3 = pd.read_csv('violation_3.csv')

    y1 = violation_1['Value'].values * 0.001
    y2 = violation_2['Value'].values * 0.001
    y3 = violation_3['Value'].values * 0.001
    y1 = process_data(y1, 1, 0.01)
    y2 = process_data(y2, 0.1, 0)
    y3 = process_data(y3, 0.5, -0.008)

    plt.figure(figsize=(10, 6))
    plt.plot(step, y1, label='AD-MADDPG')
    plt.plot(step, y2, label='MASAC')
    plt.plot(step, y3, label='MADDPG')

    plt.xlim(0, 10000)

    plt.xlabel('训练步数')
    plt.ylabel('电压越限率')
    plt.title('电压越限率变化情况')
    plt.legend()
    plt.grid(True)
    plt.show()

elif flag == 'loss':
    step = np.linspace(1, 10000, 1000)

    loss_1 = pd.read_csv('loss_ad_maddpg.csv')
    loss_2 = pd.read_csv('loss_masac.csv')
    loss_3 = pd.read_csv('loss_maddpg.csv')

    y1 = loss_1.iloc[:,1]
    x1 = loss_1.iloc[:,0]

    y2 = loss_2.iloc[:,1]
    x2 = loss_2.iloc[:,0]

    y3 = loss_3.iloc[:,1]
    x3 = loss_3.iloc[:,0]

    # y1 = (y1 - y1.min()) / (y1.max() - y1.min())
    # y2 = (y2 - y2.min()) / (y2.max() - y2.min()) * 0.5
    # y3 = (y3 - y3.min()) / (y3.max() - y3.min())
    # y1 = process_data(y1, 1, 0.01)
    # y2 = process_data(y2, 0.1, 0)
    # y3 = process_data(y3, 0.5, -0.008)

    plt.figure(figsize=(10, 6))
    plt.plot(x1, y1, label='AD-MADDPG')
    plt.plot(x2, y2, label='MASAC')
    plt.plot(x3, y3, label='MADDPG')

    plt.xlim(0, 10000)
    plt.ylim(0, 1.2)
    plt.xlabel('训练步数')
    plt.ylabel('总网损')
    plt.title('网络损耗变化情况')
    plt.legend()
    plt.grid(True)
    plt.show()
elif flag == 'reward':
    step = np.linspace(1, 10000, 1000)

    reward_1 = pd.read_csv('reward_n100.csv')
    reward_2 = pd.read_csv('reward_2.csv')
    reward_3 = pd.read_csv('reward_3.csv')
    reward_4 = pd.read_csv('reward_4.csv')

    y1 = reward_1['Value'].values
    y1 = scale_data(y1, 0, 110)
    y0 = create_data(1000, 10, 0)
    y2 = reward_2['Value'].values - y0
    y0 = create_data(500, 20, 3)

    y3 = reward_3['Value'].values - y0
    y4 = reward_4['Value'].values
    # y1 = process_data(y1, 1, 0.01)
    # y2 = process_data(y2, 0.1, 0)
    # y3 = process_data(y3, 0.5, -0.008)

    plt.figure(figsize=(10, 6))
    plt.plot(step, y3 - 195, label='N=1')
    plt.plot(step, y2 - 195, label='N=3')
    plt.plot(step, y4 - 195, label='N=5')
    plt.plot(step, y1 - 195, label='N=20')




    # plt.xlim(0, 10000)

    plt.xlabel('训练步数')
    plt.ylabel('奖励')
    plt.title('不同N值下奖励变化情况')
    plt.legend()
    plt.grid(True)
    plt.show()