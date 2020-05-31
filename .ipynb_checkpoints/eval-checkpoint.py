import sys
import torch
from configs.eval_args import args
import trains
import pandas as pd
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils.generate_csv_for_retinanet import generate_csv
if sys.argv[1]=="retinanet":
    generate_csv(args.dataset_root)
    from evals import eval_retinanet
    from data.TrashData import CSVDataset
    # Create the model
    """if args.depth == 18:
        retinanet = retinanet.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif args.depth == 34:
        retinanet = retinanet.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif args.depth == 50:
        retinanet = retinanet.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif args.depth == 101:
        retinanet = retinanet.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif args.depth == 152:
        retinanet = retinanet.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')"""
    dataset_val = CSVDataset(train_file="test_annotations.csv", class_list="classes.csv")
    retinanet = torch.load(args.model)
    mAP = eval_retinanet.evaluate(dataset_val, retinanet)
elif sys.argv[1]=="ssd":
    generate_csv(args.dataset_root)
    classes = pd.DataFrame(pd.read_csv("./classes.csv",header=None, decimal=","))
    VOC_classes = ['__background__']
    VOC_classes.extend(classes.loc[:,0].tolist())
    print(VOC_classes)
    trains.train_ssd(args.ssd_dim,len(VOC_classes))