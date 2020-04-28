import torch as t
from torch.utils import data
class CSVDataSet(data.Dataset):

    def __init__(self,annotations,classes):
        self.classes = self.extract_classes(classes)
        self.img_data = self.load_annotations(annotations)
        self.indexs = list(self.img_data.keys())

    def extract_classes(self,classes):
        file = open(classes,'r')
        result = {}
        for line in file:
            name = line.split(",")[0]
            result[name] = int(line.split(",")[1])
        return result
    def load_annotations(self, annotations):
        image_data = {}
        file = open(annotations,'r',newline='\n')
        for line in file:
            line = line.split("\n")[0]
            info = line.split(",")
            try:
                imgname, xmin,ymin,xmax,ymax,classname = info[:6]
            except ValueError:
                ValueError("the csv format is not compatible")
            if imgname not in image_data.keys():
                image_data[imgname] = []
            boxes = image_data[imgname]
            boxes.append([int(xmin),int(ymin),int(xmax),int(ymax),self.classes[classname]])
            # [[xmin, ymin, xmax, ymax, label_ind], ... ]
        return image_data

    def __len__(self):
        return len(self.indexs)