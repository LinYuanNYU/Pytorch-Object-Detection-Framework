import sys
from configs.train_args import args
import pandas as pd
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils.generate_csv_for_retinanet import generate_csv
if sys.argv[1]=="retinanet":
    from trains import train_retinanet
    generate_csv(args.dataset_root)
    train_retinanet.train_retinanet()
elif sys.argv[1]=="ssd":
    from trains import train_ssd
    generate_csv(args.dataset_root)
    classes = pd.DataFrame(pd.read_csv("./classes.csv",header=None, decimal=","))
    VOC_classes = ['__background__']
    VOC_classes.extend(classes.loc[:,0].tolist())
    train_ssd.train_ssd(args.ssd_dim,len(VOC_classes))