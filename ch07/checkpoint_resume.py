import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms
import os,sys
from torch.utils.tensorboard import SummaryWriter
base_dir = os.path.dirname(os.path.abspath(__file__))
work_dir = os.path.dirname(base_dir)
sys.path.append(work_dir)

from tools.common_tools import set_seed
from tools.my_dataset import RMBdataset
from model.lenet import LeNet


set_seed(3407)

#----------------------------------------


checkpoint_epoch = 5
batch_size = 16
max_epoch = 10
lr = 0.01

norm_mean = [0.485,0.456,0.406]
norm_std = [0.229,0.224,0.225]

data_dir = os.path.join(work_dir,'data','RMB_split_new')
train_dir = os.path.join(data_dir,'train')
valid_dir = os.path.join(data_dir,'valid')

train_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomCrop(32,padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=norm_mean,std=norm_std)
])

valid_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=norm_mean,std=norm_std)
])

train_set = RMBdataset(train_dir,train_transform)
valid_set = RMBdataset(valid_dir,valid_transform)

train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True)
valid_loader = DataLoader(valid_set,batch_size=batch_size)

#------------------------------
net = LeNet(classes=2)
net.initialize_weights()


#-------------------------
criterion = nn.CrossEntropyLoss()


#-------------------------
optimizer = optim.SGD(net.parameters(),lr=lr,momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=1)


##-----------------------------

iter_train = 0
iter_valid = 0

writer = SummaryWriter(comment='__train__interrupt__checkpoint__resume__')

checkpoint_save_path = './checkpoint__4__epoch.pkl'
checkpoint = torch.load(checkpoint_save_path)

net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
iter_train = checkpoint['iter_train']
iter_valid = checkpoint['iter_valid']

scheduler.last_epoch = start_epoch


for epoch in range(start_epoch+1,max_epoch):

    loss_train = 0.
    acc_train = 0.

    net.train()
    for i,data in enumerate(train_loader):
        iter_train += 1

        inputs,labels = data
        outputs = net(inputs)
        
        optimizer.zero_grad()
        loss_train = criterion(outputs,labels)
        loss_train.backward()

        optimizer.step()

        _,predicts = torch.max(outputs.data,1)

        correct = (predicts == labels).squeeze().sum().numpy()
        acc_train = correct/labels.size(0)

        writer.add_scalars('loss',{'train_loss':loss_train.item()},global_step=iter_train)
        writer.add_scalars('acc',{'train_acc':acc_train},global_step=iter_train)

    scheduler.step()


    if(epoch+1)%checkpoint_epoch == 0:
        checkpoint = {
            'model_state_dict':net.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'epoch':epoch,
            'iter_train':iter_train,
            'iter_valid':iter_valid
        }

        checkpoint_path = './checkpoint__{}__epoch.pkl'.format(epoch)

        torch.save(checkpoint,checkpoint_path)


    # if epoch>5:
    #     print('训练意外中断！！')
    #     break

    loss_valid = 0.
    acc_valid = 0.

    net.eval()
    with torch.no_grad():

        for j,data in enumerate(valid_loader):
            iter_valid += 1

            inputs,labels = data
            outputs = net(inputs)

            loss_valid = criterion(outputs,labels)

            _,predicts = torch.max(outputs.data,1)

            correct = (predicts == labels).squeeze().sum().numpy()

            acc_valid = correct/labels.size()

            writer.add_scalars('loss',{'valid_loss':loss_valid.item()},global_step=iter_valid*5)
            writer.add_scalars('acc',{'valid_acc':acc_valid},global_step=iter_valid*5)

writer.close()