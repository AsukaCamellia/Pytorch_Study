import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(comment='test_tensorboard')

for x in range(100):

    writer.add_scalar('y=2x',x*2,x)
    writer.add_scalar('y=pow(2,x)',2**x,x)

writer.close()


