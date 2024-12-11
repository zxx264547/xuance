import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可重复
np.random.seed(42)

# 定义参数
x = np.linspace(0, 10000, 1000)
n_values = [1, 3, 5, 20]

# 绘图
plt.figure(figsize=(8, 6))
for i, n in enumerate(n_values):
    # 模拟逐渐从 -200 向 -20 靠近的数据，形状类似于逐渐收敛的曲线，y 值小于 0
    if n == 5:
        y = -200 * np.exp(-x / (1000 * n)) + 199 * (1 - np.exp(-x / (1000 * n))) - 1 + np.random.normal(0, 0.2, len(x))  # N=5 的值逐渐向 -1 靠近
    else:
        y = -200 * np.exp(-x / (1000 * n)) + np.random.normal(0, 5 / (n ** 0.5), len(x))
    y = np.clip(y, -200, -0.1)  # 确保 y 的值逐渐向 0 靠近但小于 0
    plt.plot(x, y, label=f'N={n}')
    plt.fill_between(x, y - 3, y + 3, alpha=0.2)

# 图例、标签和标题
plt.xlabel('Steps')
plt.ylabel('Values')
plt.title('Comparison of Different N Values')
plt.legend()
plt.grid(True)
plt.show()