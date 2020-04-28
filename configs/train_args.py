import argparse
parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers()

retinanet_args = subparsers.add_parser('retinanet')
retinanet_args.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
retinanet_args.add_argument('--classes', help='Path to file containing class list (see readme)')
retinanet_args.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
retinanet_args.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
retinanet_args.add_argument('--epochs', default='50', help='epochs to train',type=int)

m2det_args = subparsers.add_parser('m2det')
m2det_args.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
m2det_args.add_argument('--classes', help='Path to file containing class list (see readme)')
m2det_args.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
parser.add_argument('-c', '--config', default='configs/m2det320_vgg16.py')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('-t', '--tensorboard', type=bool, default=False, help='Use tensorborad to show the Loss Graph')
args = parser.parse_args()
