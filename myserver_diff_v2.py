import os
import argparse, json
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from mymodels import Cifar_2NN, Cifar_CNN, Mnist_2NN, Mnist_CNN, RestNet18, MobileNetV2
from myclient_diff import ClientsGroup, client
import myPaillierPlus as paillier
import csv


# 中间参数保存路径
def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

# 保存数据到CSV文件
def save_to_csv(file_path, data, headers=None):
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if headers:
            writer.writerow(headers)  # 写入表头
        writer.writerows(data)  # 写入数据

def encrypt_vector(public_key, parameters):
    """
    将张量展平并加密
    :param public_key: Paillier公钥
    :param parameters: 需要加密的torch.Tensor
    :return: 加密后的列表和原始张量形状
    """
    shape = parameters.shape  # 记录形状信息
    parameters = parameters.flatten(0).cpu().numpy().tolist()  # 转为列表
    parameters = [public_key.encrypt(parameter) for parameter in parameters]  # 加密
    return parameters, shape


def decrypt_vector(private_key, parameters, shape):
    """
    解密并还原为张量
    :param private_key: Paillier私钥
    :param parameters: 加密后的列表
    :param shape: 原始张量的形状
    :return: 解密后的torch.Tensor
    """
    parameters = [private_key.decrypt(parameter) for parameter in parameters]  # 解密
    parameters = torch.tensor(parameters).reshape(shape)  # 还原形状为张量
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
    elif args['model_name'] == 'MobileNetV2':
        net = MobileNetV2()

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
    public_key, private_key = paillier.generate_paillier_keypair(n_length=128, precompute_h_count=args['num_comm'])

    # 测试结果数据和表头
    test_data = []
    test_headers = ['Round', 'Test Accuracy', 'Test Loss']

    # 全局迭代轮次
    for i in range(args['num_comm']):
        print("communicate round {}".format(i + 1))

        # 打乱排序，确定num_in_comm个参与方
        order = np.random.permutation(args['num_of_clients'])
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]

        sum_parameters = {}

        # 记录全局参数的shape
        parameters_shape = {}
        # 初始化 sum_parameters 为全零张量
        for key, var in global_parameters.items():
            sum_parameters[key] = torch.zeros_like(var).to(dev)
            parameters_shape[key] = var.shape

        # 可视化进度条对选中参与方local_epoch
        for client in tqdm(clients_in_comm):
            # 本地梯度下降，获取差分更新
            diff_update = myClients.clients_set[client].localUpdate(args['local_epoch'], args['batch_size'], net,
                                                                     loss_func, opti, global_parameters)

            # 没有更新说明模型已经训练好了
            if diff_update is None:
                continue
            # 对模型的差分更新进行加密
            for key in sum_parameters:
                encrypted_params, shape = encrypt_vector(public_key, (diff_update[key] * 1000).to(torch.int))
                sum_parameters[key] = {'data': encrypted_params, 'shape': shape}
        # 更新全局梯度参数
        for var in global_parameters:
            encrypted_data = sum_parameters[var]['data']
            shape = sum_parameters[var]['shape']
            decrypted_tensor = decrypt_vector(private_key, encrypted_data, shape)
            global_parameters[var] = global_parameters[var].float()
            global_parameters[var] += (decrypted_tensor.to(dev) / (num_in_comm * 1000))

        # 不进行计算图构建（无需反向传播）
        with torch.no_grad():
            # 满足评估的条件，用测试集进行数据评估
            if (i + 1) % args['val_freq'] == 0:
                # strict表示key、val严格重合才能执行（false不对齐部分默认初始化）
                net.load_state_dict(global_parameters, strict=True)
                sum_accu = 0
                num = 0
                total_loss = 0  # 用于计算测试集总损失

                # 遍历每个测试数据
                for data, label in testDataLoader:
                    # 转成gpu数据
                    data, label = data.to(dev), label.to(dev)
                    # 预测（返回结果是概率向量）
                    preds = net(data)
                    # 取最大概率label
                    loss = loss_func(preds, label)
                    total_loss += loss.item()  # 累加每批次的损失
                    preds = torch.argmax(preds, dim=1)
                    sum_accu += (preds == label).float().mean()
                    num += 1
                test_accuracy = sum_accu / num  # 计算每轮的测试集准确率
                test_loss = total_loss / num  # 计算每轮的平均损失

                # 打印并保存到CSV
                print('val_accuracy: {:.6f}, val_loss: {:.6f}'.format(test_accuracy, test_loss))
                test_data.append([i + 1, test_accuracy.item(), test_loss])



        # 根据格式和给定轮次保存参数信息
        if (i + 1) % args['save_freq'] == 0:
            torch.save(net, os.path.join(args['save_path'],
                                         '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
                                                                                                i, args['local_epoch'],
                                                                                                args['batch_size'],
                                                                                                args['learning_rate'],
                                                                                                args['num_of_clients'],
                                                                                                args['cfraction'])))
    # 保存每轮的测试结果到CSV文件
    save_to_csv(os.path.join(args['save_path'], 'test_results_en_mobileNet_diff_v2_100.csv'), test_data, test_headers)