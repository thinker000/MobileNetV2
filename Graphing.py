import time
import matplotlib.pyplot as plt
import numpy as np
import paillierPlus.mypaillier as paillier  # PHE库
from myPaillierPlus import generate_paillier_keypair  # PaillierPlus 库

# 配置参数
num_values = 93322  # Number of integers to encrypt
test_key_lengths = [128, 256, 512, 1024, 2048]  # Key lengths to test (bits)

# Store results
encryption_times_phe = [5.5256,13.0528,59.3391,359.2821,2439.0906]
decryption_times_phe = [0.8157,2.3174,8.5811,54.9597,378.3117]
encryption_times_paillierPlus = [3.3785,6.8545,29.6261,182.0883,1221.9119]
decryption_times_paillierPlus = [0.8825,2.3658,8.5602,55.0896,368.9015]

# 绘制加密时间柱状图
fig1, ax1 = plt.subplots(figsize=(10, 6))
x = np.arange(len(test_key_lengths))  # X轴位置
width = 0.35  # 柱子的宽度

# PHE加密时间
ax1.bar(x - width/2, encryption_times_phe, width, label='Paillier Encryption', color='b')
# PaillierPlus加密时间
ax1.bar(x + width/2, encryption_times_paillierPlus, width, label='PaillierPlus Encryption', color='g')

# 添加标签和标题
ax1.set_xlabel('Key Length (bits)')
ax1.set_ylabel('Time (seconds)')
ax1.set_title('Encryption Time Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(test_key_lengths)

# 添加图例
ax1.legend()

# 保存图形到本地
plt.tight_layout()
plt.savefig("encryption_time_comparison.png")  # 保存加密时间图到本地

# 绘制解密时间柱状图
fig2, ax2 = plt.subplots(figsize=(10, 6))

# PHE解密时间
ax2.bar(x - width/2, decryption_times_phe, width, label='Paillier Decryption', color='r')
# PaillierPlus解密时间
ax2.bar(x + width/2, decryption_times_paillierPlus, width, label='PaillierPlus Decryption', color='y')

# 添加标签和标题
ax2.set_xlabel('Key Length (bits)')
ax2.set_ylabel('Time (seconds)')
ax2.set_title('Decryption Time Comparison')
ax2.set_xticks(x)
ax2.set_xticklabels(test_key_lengths)

# 添加图例
ax2.legend()

# 保存图形到本地
plt.tight_layout()
plt.savefig("decryption_time_comparison.png")  # 保存解密时间图到本地
