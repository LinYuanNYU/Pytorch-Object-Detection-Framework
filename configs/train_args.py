import argparse
parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers()

retinanet_args = subparsers.add_parser('retinanet')
retinanet_args.add_argument("--dataset_root", default='data/trashV2',help='Directory path where stores the data in VOC format')
retinanet_args.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
retinanet_args.add_argument('--epochs', default='50', help='epochs to train',type=int)

ssd_args = subparsers.add_parser("ssd")
ssd_args.add_argument('--dataset_root', default='data/trashV2',
                    help='Directory path where stores the data in VOC format')
ssd_args.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
ssd_args.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
ssd_args.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
ssd_args.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
ssd_args.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
ssd_args.add_argument('--cuda', default=False,
                    help='Use CUDA to train model')
ssd_args.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
ssd_args.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
ssd_args.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
ssd_args.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
ssd_args.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
ssd_args.add_argument('--save_folder', default='/root/code/weights/ssd/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()
