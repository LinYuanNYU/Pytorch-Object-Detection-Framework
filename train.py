import sys
from configs.train_args import args
import trains
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

if sys.argv[1]=="retinanet":
    trains.train_retinanet()