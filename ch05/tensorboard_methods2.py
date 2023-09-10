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
import torchvision.models as models
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
# flag = 0 
flag = 1
if flag:
    writer = SummaryWriter(comment='__kernel__vis')

    alexnet = models.alexnet(pretrained = True)

    kernel_num = -1
    vis_max = 1

    with torch.no_grad():
        for sub_model in alexnet.modules():
            if isinstance(sub_model,nn.Conv2d):
                kernel_num += 1
                if kernel_num > vis_max:
                    break
                kernels = sub_model.weight

                c_out,c_int,k_w,k_h = tuple(kernels.shape)

                for i_out in range(c_out):
                    kernel_idx = kernels[i_out,:,:,:].unsqueeze(1)
                    kernel_grid = torchvision.utils.make_grid(kernel_idx,nrow=c_int,normalize=True,scale_each=True)
                    name = '__convlayer__split_in_channel'.format(kernel_num)
                    writer.add_image(name,kernel_grid,global_step=i_out)

                kernel_all = kernels.view(-1,3,k_h,k_w)
                kernel_all_grid = torchvision.utils.make_grid(kernel_all,nrow=8,normalize=True,scale_each=True)
                writer.add_image('__convalyer__'.format(kernel_num),kernel_all_grid,global_step=332)

        writer.close()
        
