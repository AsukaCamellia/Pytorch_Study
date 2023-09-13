import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os,sys

base_dir = os.path.dirname(os.path.abspath(__file__))
work_dir = os.path.dirname(base_dir)
log_dir = os.path.join(base_dir,'log_test')
writer = SummaryWriter(log_dir=log_dir,comment='__test__tensorboard')

for x in range(100):

    writer.add_scalar('y=2x',x*2,x)
    writer.add_scalar('y=pow(2,x)',2**x,x)
    writer.add_scalars('data/scalar_group',{'xsinx':x*np.sin(x),'xcosx':x*np.cos(x),'arctanx':np.arctan(x)},x)
    writer.add_scalar('y=3x',3*x,x)

writer.close()

#python 记录可视化数据

#存储到event file

#终端tensorboard可视化
