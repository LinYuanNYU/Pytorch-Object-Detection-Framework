import argparse
parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers()

retinanet_args = subparsers.add_parser('retinanet')
retinanet_args.add_argument("--dataset_root", default='data/trashV2',help='Directory path where stores the data in VOC format')
retinanet_args.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
retinanet_args.add_argument('--model', help='Resnet model')
args = parser.parse_args()

