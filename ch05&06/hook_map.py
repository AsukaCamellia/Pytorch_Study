import torch
import torch.nn as nn
import os,sys

base_dir = os.path.dirname(os.path.abspath(__file__))
work_dir = os.path.dirname(base_dir)
sys.path.append(work_dir)

from tools.common_tools import set_seed

set_seed(3407)


flag = 0
# flag = 1
if flag:

    w = torch.tensor([1.],requires_grad=True)
    x = torch.tensor([2.],requires_grad=True)
    a = torch.add(w,x)
    b = torch.add(w,1)
    y = torch.mul(a,b)

    a_grad = list()

    def grad_hook(grad):
        a_grad.append(grad)

    handle = a.register_hook(grad_hook)

    y.backward()

    print(w.grad,x.grad,a.grad,b.grad,y.grad)
    print(a_grad)

    handle.remove()

flag = 0
# flag = 1
if flag:

    w = torch.tensor([1.],requires_grad=True)
    x = torch.tensor([2.],requires_grad=True)
    a = torch.add(w,x)
    b = torch.add(w,1)
    y = torch.mul(a,b)


    def grad_hook(grad):
        grad *=2
        return grad*3
    
    hendle = w.register_hook(grad_hook)
    
    y.backward()

    print(w.grad)

    hendle.remove()

flag = 0
# flag = 1
if flag:

    fmap_block = list()
    input_block = list()

    class Net(nn.Module):
        def __init__(self) -> None:
            super(Net,self).__init__()
            self.conv1 = nn.Conv2d(1,2,3)
            self.pool1 = nn.MaxPool2d(2,2)

        def forward(self,x):
            x = self.conv1(x)
            x = self.pool1(x)
            return x
        
    def forward_hook(module,data_input,data_output):
            fmap_block.append(data_output)
            input_block.append(data_input)

    def forward_pre_hook(module,data_input):
            print('forward_pre_hook input: {}'.format(data_input))

    def backward_hook(module,grad_input,grad_output):
            print('backward hook input:{}'.format(grad_input))
            print('backward hook output:{}'.format(grad_output))

    net = Net()
    net.conv1.weight[0].detach().fill_(1)
    net.conv1.weight[1].detach().fill_(2)
    net.conv1.bias.data.detach().zero_()

    net.conv1.register_forward_hook(forward_hook)
    net.conv1.register_forward_pre_hook(forward_pre_hook)
    net.conv1.register_backward_hook(backward_hook)

    fake_img = torch.ones((1,1,4,4))
    output = net(fake_img)

    loss_func = nn.L1Loss()
    target = torch.randn_like(output)
    loss = loss_func(output,target)
    loss.backward()

    print('output shape',output.shape,output)
    print(fmap_block)
    print(input_block)


# flag = 0
flag = 1

if flag:
     class MLP(nn.Module):
          def __init__(self, nuearl_num,layer_num) -> None:
               super(MLP,self).__init__()
               self.linears = nn.ModuleList([nn.Linear(nuearl_num,nuearl_num,bias=False) for i in range(layer_num)])
               self.bns = nn.ModuleList([nn.BatchNorm1d(nuearl_num) for i in range(layer_num)])
               self.nuearl_num = nuearl_num

          def forward(self,x):
               for (i,linear),bn in zip(enumerate(self.linears),self.bns):
                    x = linear(x)
                    x = bn(x)
                    x = torch.relu(x)
                    if torch.isnan(x.std()):
                         print('output is nan in {} layers'.format(i))
                         break
                    print('layer:{} std:{}'.format(i,x.std().item()))
               return x   
          
          def initialize(self):
               for m in self.modules():
                    if isinstance(m,nn.Linear):
                         nn.init.kaiming_normal_(m.weight.data)

     nueral_num = 256
     layer_num = 100
     batch_size = 16

     net = MLP(nuearl_num=nueral_num,layer_num=layer_num)

     input = torch.randn((batch_size,nueral_num))

     output = net(input)
     print(output)
                         

        
    