import torch as t
import torchvision as tv
import torchvision.transforms as T
from configs.train_args import args
import torch.utils.data as torchdata
import models
import data
import utils
dataset_func = getattr(data,args.dataset)
dataset = dataset_func(args.data_root)
dataloader = torchdata.DataLoader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=2)

model_func = getattr(models,args.model)
model = model_func(class_num=133)
if t.cuda.is_available():
    model.cuda()
optims = t.optim.SGD(model.parameters(),lr=0.1)
loss_func = t.nn.CrossEntropyLoss()

for epoch in range(int(args.epochs)):
    averageloss = 0
    for step,(batch_image,batch_label) in enumerate(dataloader):
        optims.zero_grad()
        if t.cuda.is_available():
            image = t.autograd.Variable(batch_image).cuda()
            label = t.autograd.Variable(batch_label).cuda()
        else:
            image = t.autograd.Variable(batch_image)
            label = t.autograd.Variable(batch_label)
        output = model(image)
        loss = loss_func(output,label)
        loss.backward()
        averageloss += loss.data.item()
        if t.cuda.is_available() and step %20==0:
            averageloss = averageloss/20
            print("step:",step,"loss:",averageloss)
            averageloss=0
        elif not t.cuda.is_available():
            print("step:",step,"loss:",loss.data.item())
        optims.step()
    model.save(model_name="ResNet")
    print("model saved")