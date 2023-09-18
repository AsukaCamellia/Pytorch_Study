import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import os,sys
base_dir = os.path.dirname(os.path.abspath(__file__))
work_dir = os.path.dirname(base_dir)
sys.path.append(work_dir)
from tools.common_tools import set_seed
from tools.my_dataset import Antsdataset
print(work_dir)
data_dir = os.path.join(work_dir,'data','hymenoptera_data')
train_dir = os.path.join(data_dir,'train')
valid_dir = os.path.join(data_dir,'val')
net_dir = os.path.join(work_dir,'data','finetune_resnet18-5c106cde.pth')


batch_size = 64
classes = 2
max_epoch = 5
lr = 0.01
norm_mean = [0.485,0.456,0.406]
norm_std = [0.229,0.224,0.225]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

valid_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])


train_dataset = Antsdataset(train_dir,train_transform)
valid_dataset = Antsdataset(valid_dir,valid_transform)

train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
valid_loader = DataLoader(valid_dataset,batch_size=batch_size)



state_dict_load  = torch.load(net_dir)

net = models.resnet18()
net.load_state_dict(state_dict_load)
#法一 这里是冻结全部参数了  但之后替换fc之后的最后一层是可以训练的
for param in net.parameters():
    param.requires_grad = False

num_input_fc = net.fc.in_features
net.fc = nn.Linear(num_input_fc,classes)

net.to(device)


criterion = nn.CrossEntropyLoss()



#法2 小学习率
fc_params_id = list(map(id,net.fc.parameters()))
base_params = filter(lambda p:id(p) not in fc_params_id,net.parameters())
optimizer = optim.SGD([
    {'params':base_params,'lr':lr*0},
    {'params':net.fc.parameters(),'lr':lr}
],momentum=0.9)

# optimizer = optim.SGD(net.parameters(),lr=lr,momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)



iter_train = 0
iter_valid = 0
loss_train = 0.
loss_valid = 0.
acc_train = 0.
acc_valid = 0.

writer = SummaryWriter(comment='resnet_with_pretrain')

for epoch in range(max_epoch):

    net.train()

    for i,data in enumerate(train_loader):

        iter_train +=1

        inputs,labels = data
        inputs,labels = inputs.to(device),labels.to(device)
        outputs = net(inputs)

        optimizer.zero_grad()
        loss_train = criterion(outputs,labels)
        loss_train.backward()

        optimizer.step()

        _,predicts = torch.max(outputs.data,1)

        correct = (predicts == labels).squeeze().cpu().sum().numpy()

        acc_train = correct/labels.size(0)

        writer.add_scalars('loss',{'train_loss':loss_train.item()},global_step=iter_train)
        writer.add_scalars('acc',{'train_acc':acc_train},global_step=iter_train)


    scheduler.step()

    net.eval()
    with torch.no_grad():
        for j,data in enumerate(valid_loader):

            iter_valid +=1

            inputs,labels = data
            inputs,labels = inputs.to(device),labels.to(device)

            outputs = net(inputs)

            loss_valid = criterion(outputs,labels)

            _,predicts = torch.max(outputs.data,1)

            correct = (predicts == labels).squeeze().cpu().sum().numpy() 

            acc_valid = correct/labels.size(0)


            writer.add_scalars('loss',{'valid_loss':loss_valid.item()},global_step=iter_valid*5)
            writer.add_scalars('acc',{'valid_acc':acc_valid},global_step=iter_valid*5)


writer.close()
