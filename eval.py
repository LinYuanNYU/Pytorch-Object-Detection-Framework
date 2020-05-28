import sys
from configs._args import args
import trains
import pandas as pd
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils.generate_csv_for_retinanet import generate_csv
if sys.argv[1]=="retinanet":
    generate_csv(args.dataset_root)
    trains.train_retinanet()
elif sys.argv[1]=="ssd":
    generate_csv(args.dataset_root)
    classes = pd.DataFrame(pd.read_csv("./classes.csv",header=None, decimal=","))
    VOC_classes = ['__background__']
    VOC_classes.extend(classes.loc[:,0].tolist())
    print(VOC_classes)
    trains.train_ssd(args.ssd_dim,len(VOC_classes))