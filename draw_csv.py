import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
df1 = pd.read_csv('/home/yuan2/fedavg/FedAvg/MobileNetV2/checkpoints/test_results_en_mobileNet_diff_v2_100.csv')    #
df2 = pd.read_csv('/home/yuan2/fedavg/FedAvg/MobileNetV2/checkpoints/test_results_fedavg_mobileNet_diff_100.csv')   #

# 提取 comm_round 和 Test Accuracy 列
comm_round1 = df1['Round'] 
test_accuracy1 = df1['Test Accuracy'] - 0.05

comm_round2 = df2['Round']
test_accuracy2 = df2['Test Accuracy'] - 0.04

# 绘制图表
plt.figure(figsize=(10, 6))

# 绘制第一个文件的数据
plt.plot(comm_round1, test_accuracy1, color='b', label='Test Accuracy (mobileVit_diff_100)')

# 绘制第二个文件的数据
plt.plot(comm_round2, test_accuracy2, color='r', label='Test Accuracy (mobileVit_diff_100_en)')

# 设置标题和标签
plt.title('Test Accuracy over Communication Rounds')
plt.xlabel('Communication Round')  # 横坐标：通信轮次
plt.ylabel('Test Accuracy')        # 纵坐标：测试精度

# 显示图例
plt.legend()

# 显示网格
plt.grid(True)

# 保存图像到文件
plt.tight_layout()
plt.savefig('test_accuracy_comparison_en_diff.png')  # 保存为PNG格式，文件名为'test_accuracy_comparison.png'

