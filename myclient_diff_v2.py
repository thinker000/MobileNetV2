import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from mygetData import GetDataSet


import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

class client(object):
    def __init__(self, trainDataSet, dev):
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None
        self.previous_accuracy = 0.0  # 记录上一轮的精度

    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters):
        Net.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
        
        # 训练全局模型
        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)
                loss = lossFun(preds, label)
                loss.backward()
                opti.step()
                opti.zero_grad()

        # 获取本地模型参数
        local_parameters = Net.state_dict()

        # 计算差分更新 (local_parameters - global_parameters)
        diff_update = {}
        for key in local_parameters:
            diff_update[key] = local_parameters[key] - global_parameters[key]

        # 评估当前全局模型在本地数据上的效果
        current_accuracy = self.local_val(Net)

        # 如果当前精度比上一轮好，更新本地模型
        if current_accuracy > self.previous_accuracy:
            self.local_parameters = local_parameters  # 更新本地模型
            self.previous_accuracy = current_accuracy  # 更新上一轮的精度
            return diff_update  # 返回差分更新的值
        else:
            return None  # 如果精度没有提升，则不更新本地模型

    def local_val(self, Net):
        # 用本地数据评估全局模型的效果
        total_correct = 0
        total = 0
        for data, label in self.train_dl:
            data, label = data.to(self.dev), label.to(self.dev)
            preds = Net(data)
            preds = torch.argmax(preds, dim=1)
            total_correct += (preds == label).sum().item()
            total += label.size(0)

        accuracy = total_correct / total
        return accuracy



class ClientsGroup(object):
    def __init__(self, dataSetName, isIID, numOfClients, dev):
        # 数据集名称，是否独立同分布，参与方个数，GPU or CPU
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.dev = dev
        # clients_set格式为'client{i}' : Client(i)
        self.clients_set = {}

        self.test_data_loader = None

        self.dataSetBalanceAllocation()

    # 初始化CLientGroup内容
    def dataSetBalanceAllocation(self):
        mnistDataSet = GetDataSet(self.data_set_name, self.is_iid)

        # 测试集数据和标签（标签由向量转换为整型，如[0,0,1]->2）
        test_data = torch.tensor(mnistDataSet.test_data)
        test_label = torch.argmax(torch.tensor(mnistDataSet.test_label), dim=1)
        self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=100, shuffle=False)
        

        # 训练集数据和标签
        train_data = mnistDataSet.train_data
        train_label = mnistDataSet.train_label
        self.train_data_loader = DataLoader(TensorDataset(torch.tensor(train_data), torch.argmax(torch.tensor(train_label), dim=1)), batch_size=100, shuffle=False)

        shard_size = mnistDataSet.train_data_size // self.num_of_clients // 2
        shards_id = np.random.permutation(mnistDataSet.train_data_size // shard_size)
        # 初始化num_of_clients，为每个参与方分配数据
        for i in range(self.num_of_clients):
            shards_id1 = shards_id[i * 2]
            shards_id2 = shards_id[i * 2 + 1]
            data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
            local_label = np.argmax(local_label, axis=1)
            # 生成client，训练测试数据由np转为tensor
            someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
            self.clients_set['client{}'.format(i)] = someone

# 测试ClientGroup
# if __name__=="__main__":
#     MyClients = ClientsGroup('mnist', True, 100, 1)
#     print(MyClients.clients_set['client10'].train_ds[0:100])
#     print(MyClients.clients_set['client11'].train_ds[400:500])


