import torch as t
import os
import time
class BaseModule(t.nn.Module):
    def __init__(self):
        super(BaseModule,self).__init__()
        self.model_name = str(type(self))
    def save(self,name=None,model_name=""):
        if name==None:
            prefix = os.path.join("pths/"+model_name+"_")
            savename = time.strftime(prefix+'%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), savename)
    def load(self,path,map_location=None):
        self.load_state_dict(t.load(path,map_location=map_location))