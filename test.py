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

# Evaluate the model on the test data
def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    return test_loss, test_accuracy


def resize_transform(aux):
    return transforms.Resize(aux[None, None,:].shape[1:])[0,0]

def gen_gradcams(data_loader, parse):
    
    img_idx = 0

    for batch_idx, (data, target) in enumerate(test_loader):
        res = torch.zeros((data_loader.size, args.image_size, args.image_size))
        targets = torch.zeros((data_loader.size,))

        for ix in range(data.shape[0]):
          img = data[ix]
          resize_transform = 
          aux = get_gradcam(model, img, target[ix])
          proyected_grad_cam = resize_transform(aux)
          res[img_idx] = proyected_grad_cam
          target[img_idx] = target[ix]

          img_idx += 1

          
# Load the model
model = LeNet([args.image_size, args.image_size, 3], 512)
model.load_state_dict(torch.load(args.model_destination + 'model.pt'))
model.eval()

test_loss, test_accuracy = test(model, test_loader)
print('Epoch: {}/{} Test Loss: {:.4f} Test Accuracy: {:.2f}%'.format(
      epoch, epochs, test_loss, test_accuracy))



