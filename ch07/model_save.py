import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models

class simplenet(nn.Module):

    def __init__(self, classes) -> None:
        super(simplenet,self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3,6,5),
            nn.BatchNorm2d(6),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

    def forward(self,x):
        x = self.features(x)
        return x
    
    def initialize(self):
        for m in self.parameters():
            m.data.fill_(2023)


net = simplenet(classes=2)

model_path = './model.pkl'
state_dict_path = './state_dict_path.pkl'

print('before',net.features[0].weight[0,...])

net.initialize()

print('after',net.features[0].weight[0,...])

torch.save(net,model_path)
state_dict = net.state_dict()
torch.save(state_dict,state_dict_path)
