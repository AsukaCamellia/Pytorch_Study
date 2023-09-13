import torch
import torch.nn as nn
import numpy as np

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

net_whole = torch.load('./model.pkl')
print(net_whole.features[0].weight[0,...])

print(net.features[0].weight[0,...])
state_dict = torch.load('./state_dict_path.pkl')
net.load_state_dict(state_dict)
print(net.features[0].weight[0,...])



