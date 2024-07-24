import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from resnet import ResNet18

class LocalDevice:
    def __init__(self, id, data_type, device_type, user_type, set_length, url) -> None:
        # id号
        self.id = id
        # 数据集
        self.data_type = data_type
        # 设备类型
        self.device_type = device_type
        # 用户类型
        self.user_type = user_type
        # 连接的地址
        self.url = url
        # 数据集长度
        self.set_length = set_length

        if data_type == 'cifar10':
            self.model = ResNet18()
            transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            # device type = 0 ，为客户端
            if self.device_type == 0:
                trainset = torchvision.datasets.CIFAR10(root='./data/cifar', train=True, download=True, transform=transform_train)
                # 按照set_length长度随机取子集
                indices = np.random.choice(len(trainset), self.set_length, replace=False)
                self.trainset = torch.utils.data.Subset(trainset, indices)
                self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=128, shuffle=True, num_workers=0)
            # device type = 1 ，为边缘服务器
            elif self.device_type == 1:
                transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
                self.testset = torchvision.datasets.CIFAR10(root='./data/cifar', train=False, download=True, transform=transform_test)
                self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=100, shuffle=False, num_workers=0)

            self.loss_fn=torch.nn.CrossEntropyLoss()
            self.lr = 0.1
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

    def set_trainset(self, new_set):
        self.trainset = new_set
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=128, shuffle=True, num_workers=0)

    def express(self):
        print(f'device id : {self.id} | dataset : {self.data_type} | device type : {self.device_type} | user type : {self.user_type} | url : {self.url} | set length : {len(self.trainset)}')

        


    