import torchvision as tv
import os
import utils
import torchvision.transforms as T
import torch as t
class DogData(tv.datasets.ImageFolder):
    def __init__(self,root,train=True,transforms=None,inputShape=224):
        if train:
            newroot = os.path.join(root,"train")
        else:
            newroot = os.path.join(root,"test")
        if transforms==None:
            transforms = T.Compose(
                        [T.Resize([224,224]),
                         T.ToTensor(),
                         ])
        super(DogData,self).__init__(root=newroot,transform=transforms)
        self.input_shape = inputShape
        #self.mean = self.cal_mean()

    def cal_mean(self):
        meanTensor = t.zeros([3,self.input_shape,self.input_shape])
        img2Tensor = T.ToTensor()
        for img,_ in self:
            meanTensor +=img2Tensor(img)
            print(meanTensor)
        meanTensor = meanTensor/len(self)
        print(meanTensor)