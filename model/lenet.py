import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):

    def __init__(self,classes) -> None:
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.maxpool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.maxpool2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,classes)

    def forward(self,x):
        out = F.relu(self.conv1(x))
        out = self.maxpool1(out)
        out = F.relu(self.conv2(out))
        out = self.maxpool2(out)
        out = out.view(out.size(0),-1)
        #out.size(0)就是batch_size,-1就是把三维展成1维，这里就是400
        #out = out.view(out.size(0),400)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight.data,0,0.1)
                m.bias.data.zero_()


class LeNetSequential(nn.Module):

    def __init__(self, classes):
        super(LeNetSequential,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,6,5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(6,16,5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16*5*5,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,classes),
        )

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)

        return x