import torch as t
import torchvision as tv
import torch.nn as nn
import models.BaseModule as BaseModule
import torchvision.models as models

class ResNet(BaseModule):
    def __init__(self,class_num):
        super(BaseModule,self).__init__()
        self.cnn = models.resnet50(pretrained=True)
        self.fc = nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(1000,class_num),
                        nn.Softmax()
        )
    def forward(self,x):
        x = self.cnn(x)
        x = self.fc(x)
        return x


