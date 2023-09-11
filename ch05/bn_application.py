import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import torchvision.transforms as transforms
import torch.optim as optim
import sys,os
from torch.utils.data import DataLoader,Dataset
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

base_dir = os.path.dirname(os.path.abspath(__file__))
work_dir = os.path.dirname(base_dir)
sys.path.append(work_dir)

from tools.common_tools import set_seed
from tools.my_dataset import RMBdataset


set_seed(3407)

class lenet(nn.Module):
    def __init__(self,classes) -> None:
        super(lenet,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,6,5),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6,16,5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16*5*5,120),
            nn.BatchNorm1d(120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,classes)
        )

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight.data,0,0.1)
                m.bias.data.zero_()
                

batch_size = 16
max_epoch = 20
lr = 0.01

data_dir = os.path.join(work_dir,'data','RMB_split_new')
train_dir = os.path.join(data_dir,'train')
valid_dir = os.path.join(data_dir,'valid')



norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomCrop(32,padding=4),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean,norm_std)
])

valid_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean,norm_std)
])

train_data = RMBdataset(data_dir=train_dir,transform=train_transform)
valid_data = RMBdataset(data_dir=valid_dir,transform=valid_transform)

train_loader = DataLoader(train_data,batch_size,shuffle=True)
valid_loader = DataLoader(valid_data,batch_size)

net = lenet(classes=2)
net.initialize()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(),lr=lr,momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=10,gamma=0.1)

train_curve = list()
valid_curve = list()
iter_count = 0
iter_val_count = 0
writer = SummaryWriter(comment='__BatchNorm__')

for epoch in range(max_epoch):

    net.train()


    for i,data in enumerate(train_loader):
        iter_count +=1

        inputs,labels = data
        outputs = net(inputs)


        optimizer.zero_grad()        
        loss = criterion(outputs,labels)
        loss.backward()

        optimizer.step()

        train_curve.append(loss.item())
        _,predicts = torch.max(outputs.data,1)

        correct = (predicts == labels).squeeze().sum().numpy()
        acc = correct/labels.size(0)

        writer.add_scalars('loss',{'train':loss.item()},iter_count)
        writer.add_scalars('acc',{'train':acc},iter_count)


    scheduler.step()
    net.eval()
    with torch.no_grad():
        for j,data in enumerate(valid_loader):
            iter_val_count += 1

            inputs,labels = data
            outputs = net(inputs)

            loss_val = criterion(outputs,labels)

            valid_curve.append(loss.item())
            _,predicts = torch.max(outputs.data,1)

            correct_val = (predicts == labels).squeeze().sum().numpy()
            acc_val = correct_val/labels.size(0)

            writer.add_scalars('loss',{'valid':loss_val.item()},iter_val_count*5)
            writer.add_scalars('acc',{'valid':acc_val},iter_val_count*5)

writer.close()







