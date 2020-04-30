import sys
from configs.train_args import args
import trains
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils.generate_csv_for_retinanet import generate_csv
if sys.argv[1]=="retinanet":
    generate_csv(args.data_root)
    trains.train_retinanet()
elif sys.argv[1]=="ssd":
    trains.