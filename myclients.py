
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from mygetData import GetDataSet


class client(object):
    def __init__(self, trainDataSet, dev):
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None

    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters):
        Net.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
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


        return local_parameters  # 返回本地模型参数

    def local_val(self):
        pass


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
        if self.data_set_name == 'mnist':
            mnistDataSet = GetDataSet(self.data_set_name, self.is_iid)

            test_data = torch.tensor(mnistDataSet.test_data)
            test_label = torch.argmax(torch.tensor(mnistDataSet.test_label), dim=1)
            self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=100, shuffle=False)

            train_data = mnistDataSet.train_data
            train_label = mnistDataSet.train_label
            self.train_data_loader = DataLoader(TensorDataset(torch.tensor(train_data), torch.argmax(torch.tensor(train_label), dim=1)), batch_size=100, shuffle=False)

        elif self.data_set_name == 'cifar10':
            cifar10DataSet = GetDataSet(self.data_set_name, self.is_iid)

            test_data = torch.tensor(cifar10DataSet.test_data)
            test_label = torch.tensor(cifar10DataSet.test_label)
            self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=100, shuffle=False)

            train_data = cifar10DataSet.train_data
            train_label = cifar10DataSet.train_label
            self.train_data_loader = DataLoader(TensorDataset(torch.tensor(train_data), torch.tensor(train_label)), batch_size=100, shuffle=False)

        # 分配数据给每个client
        shard_size = len(train_data) // self.num_of_clients // 2
        shards_id = np.random.permutation(len(train_data) // shard_size)
        
        for i in range(self.num_of_clients):
            shards_id1 = shards_id[i * 2]
            shards_id2 = shards_id[i * 2 + 1]
            data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
            
            # 如果是 CIFAR-10，标签本身已经是数字，不需要进行 one-hot 编码
            someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
            self.clients_set['client{}'.format(i)] = someone


# 测试ClientGroup
if __name__ == "__main__":
    MyClients = ClientsGroup('cifar10', True, 100, 1)
    print(MyClients.clients_set['client10'].train_ds[0:100])