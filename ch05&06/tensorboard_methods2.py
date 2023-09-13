import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import os,sys
base_dir = os.path.dirname(os.path.abspath(__file__))
work_dir = os.path.dirname(base_dir)
sys.path.append(work_dir)
from tools.common_tools import set_seed
from tools.my_dataset import RMBdataset 
from model.lenet import LeNet
import torchvision.models as models
from PIL import Image
data_dir = os.path.join(work_dir,'data','RMB_split_new')
train_dir = os.path.join(data_dir,'train')



flag = 0
# flag = 1
if flag:
    
    writer = SummaryWriter(comment='__img__vis__')
    fake_img = torch.randn((3,512,512))
    writer.add_image('fake_img',fake_img,1)


    fake_img = torch.ones((3,512,512))
    writer.add_image('fake_img',fake_img,2)


    fake_img = torch.ones((3,512,512))
    writer.add_image('fake_img',fake_img*1.1,3)


    fake_img = torch.rand((512,512))
    writer.add_image('fake_img',fake_img,4,dataformats='HW')


    fake_img = torch.rand((512,512,3))
    writer.add_image('fake_img',fake_img,5,dataformats='HWC')

    writer.close()


#检查我们的训练数据输入是否正确
flag = 0
# flag = 1
if flag:
    writer = SummaryWriter(comment='__make__grid__vis')

    norm_mean = [0.485,0.456,0.406]
    norm_std = [0.229,0.224,0.225]

    train_transform = transforms.Compose([
        transforms.Resize((32,64)),
        transforms.ToTensor()
    ])

    train_dataset = RMBdataset(train_dir,train_transform)
    train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True)
    traindata_batch,trainlabel_batch = next(iter(train_loader))

    # img_grid = torchvision.utils.make_grid(traindata_batch,nrow=4,normalize=True,scale_each=True)
    img_grid = torchvision.utils.make_grid(traindata_batch,nrow=4,normalize=True,scale_each=False)

    # writer.add_image('make_grid_scale_each = True',img_grid,1)
    writer.add_image('make_grid_scale_each = False',img_grid,1)

    writer.close()

#---------------kernel vis
flag = 0 
# flag = 1
if flag:
    writer = SummaryWriter(comment='__kernel__vis__')

    net = models.alexnet(pretrained = True)

    layer_num = -1
    vis_maxnum = 1

    for sub_model in net.modules():
        if isinstance(sub_model,nn.Conv2d):
            layer_num += 1
            if layer_num>vis_maxnum:
                break
            kernels = sub_model.weight

            #c_out实际上就是这一层卷积核的个数
            c_out,c_int,k_h,k_w = tuple(kernels.shape)

            for i in range(c_out):
                kernels_1d = kernels[i,:,:,:].unsqueeze(1)
                kernel_grid = torchvision.utils.make_grid(kernels_1d,nrow=c_int,normalize=True,scale_each=True)
                writer.add_image('{}__convlayer__kernel_split_in_channel'.format(layer_num),kernel_grid,global_step=i)


            kernels_all = kernels.view(-1,3,k_h,k_w)
            kernels_all_grid = torchvision.utils.make_grid(kernels_all,nrow=8,normalize=True,scale_each=True)
            writer.add_image('{}__convlayer_kernel'.format(layer_num),kernels_all_grid,global_step=layer_num)

            print(kernels.shape)

    writer.close()


flag = 0
# flag = 1
if flag:
    writer = SummaryWriter(comment='__feature__map__vis')

    img_path = os.path.join(base_dir,'lena.png')

    normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]

    img_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(normMean,normStd)
    ])

    img_pil = Image.open(img_path).convert('RGB')

    if img_transform is not None:
        img_tensor = img_transform(img_pil)

    img_tensor = img_tensor.unsqueeze(0)

    net = models.alexnet(pretrained = True)

    convlayer1 = net.features[0]

    feature_map1 = convlayer1(img_tensor)

    feature_map1.transpose_(0,1)

    fea_map_grid = torchvision.utils.make_grid(feature_map1,nrow=8,normalize=True,scale_each=True)

    writer.add_image('__feature__map',fea_map_grid,global_step=0)

    writer.close()

# flag = 0
flag = 1

if flag:

    writer = SummaryWriter(comment='__graph__vis__')

    lenet = LeNet(classes=2)

    fake_img = torch.randn((1,3,32,32))

    writer.add_graph(lenet,fake_img)

    writer.close()