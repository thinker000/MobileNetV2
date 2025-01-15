import os
import argparse
import json
from tqdm import tqdm
import numpy as np
import torch
import time
import torch.nn.functional as F
from torch import optim
from mymodels import Cifar_2NN, Cifar_CNN, Mnist_2NN, Mnist_CNN, RestNet18
from myclients import ClientsGroup, client
import matplotlib.pyplot as plt  # 导入matplotlib
import csv  # 用于保存数据到CSV文件


# 中间参数保存路径
def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

# 保存训练数据到CSV文件
def save_to_csv(file_path, data, headers=None):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        if headers:
            writer.writerow(headers)  # 写入表头
        writer.writerows(data)  # 写入数据

if __name__ == "__main__":

    # 定义解析器
    parser = argparse.ArgumentParser(description='FedAvg')
    parser.add_argument('-c', '--conf', dest='conf')
    arg = parser.parse_args()

    # 解析器解析json文件
    with open(arg.conf, 'r') as f:
        args = json.load(f)

    # 创建中间参数保存目录
    test_mkdir(args['save_path'])

    # 使用gpu or cpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 定义使用模型(全连接或卷积)
    net = None
    if args['model_name'] == 'mnist_2nn':
        net = Mnist_2NN()
    elif args['model_name'] == 'mnist_cnn':
        net = Mnist_CNN()
    elif args['model_name'] == 'cifar_cnn':
        net = Cifar_CNN()
    elif args['model_name'] == 'resnet18':
        net = RestNet18()
    elif args['model_name'] == 'cifar_2nn':
        net = Cifar_2NN()

    # 如果gpu设备不止一个，并行计算
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
    net = net.to(dev)

    # 定义损失函数和优化器
    loss_func = F.cross_entropy
    opti = optim.Adam(net.parameters(), lr=args['learning_rate'])

    # 定义数据集
    type = args['type']

    # 定义多个参与方，导入训练、测试数据集
    myClients = ClientsGroup(type, args['IID'], args['num_of_clients'], dev)
    testDataLoader = myClients.test_data_loader
    trainDataLoader = myClients.train_data_loader

    # 每轮迭代的参与方个数
    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

    # 初始化全局参数
    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    # 定义保存准确率、损失、通信时间的列表
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []  # 用于记录每轮的测试集准确率
    test_losses = []  # 用于记录每轮的测试集损失
    comm_times = []  # 用于记录每轮的通信时间

    # 保存测试过程中的数据到CSV文件
    test_data = []
    test_headers = ['Round', 'Test Accuracy', 'Test Loss', 'Comm Time']

    # 全局迭代轮次
    for i in range(args['num_comm']):
        print("communicate round {}".format(i+1))
        start_time = time.time() 
        # 打乱排序，确定num_in_comm个参与方
        order = np.random.permutation(args['num_of_clients'])
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]

        sum_parameters = None

        # 记录全局参数的shape
        parameters_shape = None

        # 可视化进度条对选中参与方local_epoch
        for client in tqdm(clients_in_comm):
            # 本地梯度下降
            local_parameters = myClients.clients_set[client].localUpdate(args['local_epoch'], args['batch_size'], net,
                                                                         loss_func, opti, global_parameters)
            
            # 初始化sum_parameters
            if sum_parameters is None:
                sum_parameters = {}
                parameters_shape = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone()
                    parameters_shape[key] = var.shape

            else:
                for key in sum_parameters:
                    sum_parameters[key] = sum_parameters[key] + local_parameters[key]
        
        end_time = time.time()
        comm_time = end_time - start_time
        comm_times.append(comm_time)  # 记录每轮的通信时间
        print(f"communication round {i+1} time: {comm_time:.6f}s")
        
        # 更新全局梯度参数
        for var in global_parameters:
            sum_parameters[var] = torch.reshape(torch.Tensor(sum_parameters[var]), parameters_shape[var])
            global_parameters[var] = (sum_parameters[var].to(dev) / num_in_comm)
        
        # 不进行计算图构建（无需反向传播）
        with torch.no_grad():
            # 满足评估的条件，用测试集进行数据评估
            if (i + 1) % args['val_freq'] == 0:
                # strict表示key、val严格重合才能执行（false不对齐部分默认初始化）
                net.load_state_dict(global_parameters, strict=True)

                # 测试集评估
                sum_accu = 0
                num = 0
                total_loss = 0  # 用于计算测试集总损失
                for data, label in testDataLoader:
                    data, label = data.to(dev), label.to(dev)
                    preds = net(data)
                    loss = loss_func(preds, label)
                    total_loss += loss.item()  # 累加每批次的损失
                    preds = torch.argmax(preds, dim=1)               
                    sum_accu += (preds == label).float().mean()
                    num += 1
                test_accuracy = sum_accu / num  # 计算每轮的测试集准确率
                test_accuracies.append(test_accuracy.item())  # 记录每轮测试集准确率
                test_loss = total_loss / num  # 计算每轮的平均损失
                test_losses.append(test_loss)  # 记录每轮测试集损失
                print('test_accuracy: {:.6f}, test_loss: {:.6f}'.format(test_accuracy, test_loss))

                # 保存测试数据到test_data列表
                test_data.append([i+1, test_accuracy.item(), test_loss, comm_time])

        # 根据格式和给定轮次保存参数信息
        if (i + 1) % args['save_freq'] == 0:
            torch.save(net, os.path.join(args['save_path'],
                                         '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
                                                                                                i, args['local_epoch'],
                                                                                                args['batch_size'],
                                                                                                args['learning_rate'],
                                                                                                args['num_of_clients'],
                                                                                                args['cfraction'])))
    
    # 保存测试结果到CSV文件
    save_to_csv(os.path.join(args['save_path'], 'test_results_noEncryptServer__noen_nodiff.csv'), test_data, test_headers)

    # 绘制测试集损失图表
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss')
    plt.xlabel('Communication Round')
    plt.ylabel('Loss')
    plt.title('Test Loss over Communication Rounds')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args['save_path'], 'test_loss_plot_noen_nodiff.png'))
    
    # 绘制测试集准确率图表
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Communication Round')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy over Communication Rounds')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args['save_path'], 'test_accuracy_plot_noen_nodiff.png'))
