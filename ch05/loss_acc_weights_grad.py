import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import matplotlib.pyplot as plt
import os,sys

base_dir = os.path.dirname(os.path.abspath(__file__))
work_dir = os.path.dirname(base_dir)
sys.path.append(work_dir)

from tools.common_tools import set_seed
from model.lenet import LeNet
from tools.my_dataset import RMBdataset

set_seed(3407)

batch_size = 8
lr = 0.01
max_epoch = 10

data_dir = os.path.join(work_dir,'data','RMB_split_new')
train_dir = os.path.join(data_dir,'train')
test_dir = os.path.join(data_dir,'test')
valid_dir = os.path.join(data_dir,'valid')

norm_mean = [0.485,0.456,0.406]
norm_std = [0.229,0.224,0.225]

train_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomCrop(32,padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=norm_mean,std=norm_std),
])
valid_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=norm_mean,std=norm_std),
])

#构建dataset
train_data = RMBdataset(train_dir,train_transform)
valid_data = RMBdataset(valid_dir,valid_transform)

#构建dataloader
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
valid_data = DataLoader(valid_data,batch_size=batch_size)

net = LeNet(classes=2)
net.initialize_weights()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(),lr = lr,momentum=0.9)
scheduler =optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)

train_curve = list()
valid_curve = list()

for epoch in range(max_epoch):

    for i,data in enumerate(train_loader):

        #forward
        inputs,labels = data
        outputs = net(inputs)

        #backward
        optimizer.zero_grad()
        loss = criterion(outputs,labels)
        loss.backward()

        #update weights
        optimizer.step()

        _,predict = torch.max(outputs.data,1)

        correct = (predict == labels).squeeze().sum().numpy()
        acc = correct/labels.size()

        print(epoch,'  ',i,'  ',loss.item(),'  ',acc)

