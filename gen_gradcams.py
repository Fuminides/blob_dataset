import torch
import torch.nn as nn
from model_grad_cam import LeNet
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from parser import gen_parser
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from gradcam import get_gradcam
from blob_dataloader import BlobDataset

# Parse the arguments
parser = gen_parser()
args = parser.parse_args()

# Load the data
if args.dataset == 'mnist':
    test_dataset = datasets.MNIST(root='.', train=False, transform=transforms.ToTensor(), download=True)
    train_dataset = datasets.MNIST(root='.', train=True, transform=transforms.ToTensor(), download=True)
elif args.dataset == 'cifar10':
    test_dataset = datasets.CIFAR10('data', train=False, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))
                         ]))
    train_dataset = datasets.CIFAR10('data', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))
                         ]))
elif args.dataset == 'blob':
    dataset = BlobDataset('trials/', train=False)


print('Generating the gradcam maps...')
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

gen_gradcams(train_loader, args)
gen_gradcams(test_loader, args)


