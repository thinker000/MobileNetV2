import numpy as np
from torchvision import datasets, transforms


class GetDataSet(object):
    def __init__(self, dataSetName, isIID):
        self.name = dataSetName
        self.train_data = None
        self.train_label = None
        self.train_data_size = None
        self.test_data = None
        self.test_label = None
        self.test_data_size = None

        self._index_in_train_epoch = 0

        if self.name == 'mnist':
            self.mnistDataSetConstruct(isIID)
        elif self.name == 'cifar10':
            self.cifar10DataSetConstruct(isIID)

    def mnistDataSetConstruct(self, isIID):
        from torchvision import datasets
        data_dir = './data/mnist'

        transform = transforms.Compose([transforms.ToTensor()])
        mnist_data = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        mnist_test_data = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

        self.train_data = np.array(mnist_data.data)  # MNIST train data
        self.train_label = np.array(mnist_data.targets)  # MNIST train labels
        self.test_data = np.array(mnist_test_data.data)  # MNIST test data
        self.test_label = np.array(mnist_test_data.targets)  # MNIST test labels

        self.train_data_size = len(self.train_data)
        self.test_data_size = len(self.test_data)

        if isIID:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = self.train_data[order]
            self.train_label = self.train_label[order]
        else:
            # 非IID分配
            self.nonIIDDataAllocation()

    def nonIIDDataAllocation(self):
        # 非IID分配：按类别进行数据分配
        num_classes = 10  # MNIST的类别数
        num_clients = 3  # 客户端数目
        num_samples_per_class = self.train_data_size // num_classes

        # 按照类别进行划分
        data_by_class = {}
        for class_id in range(num_classes):
            data_by_class[class_id] = {
                'data': self.train_data[self.train_label == class_id],
                'label': self.train_label[self.train_label == class_id]
            }

        # 为每个客户端分配数据：每个客户端的样本数是不均匀的
        clients_data = {i: [] for i in range(num_clients)}

        # 假设我们把每个类别的数据随机分配给客户端
        for class_id in range(num_classes):
            class_data = data_by_class[class_id]
            num_samples = len(class_data['data'])
            client_indices = np.random.choice(num_clients, num_samples, replace=True)  # 随机分配给多个客户端
            for i, client_idx in enumerate(client_indices):
                clients_data[client_idx].append((class_data['data'][i], class_data['label'][i]))

        # 将分配的数据合并回去
        self.train_data = []
        self.train_label = []
        for client_data in clients_data.values():
            client_data = np.array(client_data)
            self.train_data.append(client_data[:, 0])
            self.train_label.append(client_data[:, 1])

        self.train_data = np.vstack(self.train_data)
        self.train_label = np.hstack(self.train_label)

    def cifar10DataSetConstruct(self, isIID):
        data_dir = './data/cifar10'  # CIFAR-10 数据存放目录

        # 转换操作（调整大小、转换为tensor、归一化）
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # 加载训练集和测试集
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)

        self.train_data = train_dataset.data
        self.train_label = np.array(train_dataset.targets)
        self.test_data = test_dataset.data
        self.test_label = np.array(test_dataset.targets)

        self.train_data_size = len(self.train_data)
        self.test_data_size = len(self.test_data)

        # 如果是 IID，则打乱数据
        if isIID:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = self.train_data[order]
            self.train_label = self.train_label[order]
        else:
            # 非IID分配
            self.nonIIDDataAllocation()

    def nonIIDDataAllocation(self):
        # 非IID分配：按类别进行数据分配
        num_classes = 10  # CIFAR-10的类别数
        num_clients = 100  # 客户端数目
        num_samples_per_class = self.train_data_size // num_classes

        # 按照类别进行划分
        data_by_class = {}
        for class_id in range(num_classes):
            data_by_class[class_id] = {
                'data': self.train_data[self.train_label == class_id],
                'label': self.train_label[self.train_label == class_id]
            }

        # 为每个客户端分配数据：每个客户端的样本数是不均匀的
        clients_data = {i: [] for i in range(num_clients)}

        # 假设我们把每个类别的数据随机分配给客户端
        for class_id in range(num_classes):
            class_data = data_by_class[class_id]
            num_samples = len(class_data['data'])
            client_indices = np.random.choice(num_clients, num_samples, replace=True)  # 随机分配给多个客户端
            for i, client_idx in enumerate(client_indices):
                clients_data[client_idx].append((class_data['data'][i], class_data['label'][i]))

        # 将分配的数据合并回去
        self.train_data = []
        self.train_label = []
        for client_data in clients_data.values():
            client_data = np.array(client_data)
            self.train_data.append(client_data[:, 0])
            self.train_label.append(client_data[:, 1])

        self.train_data = np.vstack(self.train_data)
        self.train_label = np.hstack(self.train_label)
