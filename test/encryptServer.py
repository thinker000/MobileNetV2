import os
import argparse, json
from tqdm import tqdm
import numpy as np
import torch
import time
import torch.nn.functional as F
from torch import optim
from mymodels import Cifar_2NN, Cifar_CNN, Mnist_2NN, Mnist_CNN, RestNet18
from myclients import ClientsGroup, client
from phe import paillier
import matplotlib.pyplot as plt  # 导入matplotlib


# 中间参数保存路径
def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

# torch转list加密
def encrypt_vector(public_key, parameters):
    global global_encrypt_time
    start_time = time.time()
    parameters = parameters.flatten(0).cpu().numpy().tolist()
    parameters = [public_key.encrypt(parameter) for parameter in parameters]
    end_time = time.time()
    encrypt_time = end_time - start_time
    print(f"encrypt_time: {encrypt_time:.6f}s")
    return parameters

# list 解密
def decrypt_vector(private_key, parameters):
    global global_decrypt_time
    start_time = time.time()
    parameters = [private_key.decrypt(parameter) for parameter in parameters]
    end_time = time.time()
    decrypt_time = end_time - start_time
    print(f"decrypt_time: {decrypt_time:.6f}s")
    return parameters

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

    # 定义使用模型(全连接 or 简单卷积)
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

    # 生成密钥
    public_key, private_key = paillier.generate_paillier_keypair(n_length=1024)

    # 定义保存准确率和损失的列表
    train_accuracies = []
    val_accuracies = []
    train_losses = []  # 用于保存训练损失
    val_losses = []    # 用于保存验证损失
    comm_times = []    # 用于存储每次通信的时间

    # 全局迭代轮次
    for i in range(args['num_comm']):
        print("communicate round {}".format(i + 1))
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
                    sum_parameters[key] = encrypt_vector(public_key, sum_parameters[key])
            else:
                for key in sum_parameters:
                    sum_parameters[key] = np.add(sum_parameters[key], encrypt_vector(public_key, local_parameters[key]))

        # 计算通信时间
        end_time = time.time()
        comm_time = end_time - start_time
        comm_times.append(comm_time)  # 记录通信时间
        print(f"communication time: {comm_time:.6f}s")

        # 更新全局梯度参数
        for var in global_parameters:
            sum_parameters[var] = decrypt_vector(private_key, sum_parameters[var])
            sum_parameters[var] = torch.reshape(torch.Tensor(sum_parameters[var]), parameters_shape[var])
            global_parameters[var] = (sum_parameters[var].to(dev) / num_in_comm)

        # 更新模型参数
        net.load_state_dict(global_parameters, strict=True)
    
        # 不进行计算图构建（无需反向传播）
        with torch.no_grad():
            # 满足评估的条件，用测试集进行数据评估
            if (i + 1) % args['val_freq'] == 0:
                # strict表示key、val严格重合才能执行（false不对齐部分默认初始化）
                net.load_state_dict(global_parameters, strict=True)

                # 测试集评估
                sum_accu = 0
                sum_loss = 0  # 记录验证集的损失
                num = 0
                for data, label in testDataLoader:
                    data, label = data.to(dev), label.to(dev)
                    preds = net(data)
                    loss = loss_func(preds, label)
                    sum_loss += loss.item()
                    preds = torch.argmax(preds, dim=1)               
                    sum_accu += (preds == label).float().mean()
                    num += 1
                val_accuracy = sum_accu / num
                val_losses.append(sum_loss / num)  # 记录验证集的平均损失
                val_accuracies.append(val_accuracy.item())
                print('val_accuracy: {:.6f}'.format(val_accuracy))

                # 训练集评估
                sum_accu = 0
                sum_loss = 0  # 记录训练集的损失
                num = 0
                for data, label in trainDataLoader:
                    data, label = data.to(dev), label.to(dev)
                    preds = net(data)
                    loss = loss_func(preds, label)
                    sum_loss += loss.item()
                    preds = torch.argmax(preds, dim=1)               
                    sum_accu += (preds == label).float().mean()
                    num += 1
                train_accuracy = sum_accu / num
                train_losses.append(sum_loss / num)  # 记录训练集的平均损失
                train_accuracies.append(train_accuracy.item())
                print('train_accuracy: {:.6f}'.format(train_accuracy))

        # 根据格式和给定轮次保存参数信息
        if (i + 1) % args['save_freq'] == 0:
            torch.save(net, os.path.join(args['save_path'],
                                         '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
                                                                                                i, args['local_epoch'],
                                                                                                args['batch_size'],
                                                                                                args['learning_rate'],
                                                                                                args['num_of_clients'],
                                                                                                args['cfraction'])))
    
    # 绘制训练和验证准确率曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Communication Round')
    plt.ylabel('Accuracy')
    plt.title('Train and Validation Accuracy over Communication Rounds')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args['save_path'], 'accuracy_plot_enyes50.png'))

    # 绘制训练和验证损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Communication Round')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss over Communication Rounds')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args['save_path'], 'loss_plot_enyes50.png'))

    # 绘制通信时间曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(comm_times) + 1), comm_times, label='Communication Time')
    plt.xlabel('Communication Round')
    plt.ylabel('Time (s)')
    plt.title('Communication Time per Round')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args['save_path'], 'comm_time_plot_enyes50.png'))
