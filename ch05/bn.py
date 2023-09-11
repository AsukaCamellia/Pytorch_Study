import torch
import torch.nn as nn

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
                         
