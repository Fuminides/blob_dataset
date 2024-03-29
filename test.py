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
    test_dataset = datasets.MNIST(root='.', train=False, transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor()]), download=True)
elif args.dataset == 'cifar10':
    test_dataset = datasets.CIFAR10('data', train=False, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))
                         ]))
 
elif args.dataset == 'blob':
    test_dataset = BlobDataset(args.dataset_path, train=False, transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor()]))

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Evaluate the model on the test data
def test(model, test_loader):
    model.eval()

    correct_number = 0
    correct_size = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output_number, output_size = model(data)
            target_list = []
            for target_x in target:
                target_x = target_x.to(device)
                target_list.append(target_x)
            
            pred_number = torch.argmax(output_number, dim=1)
            pred_size = torch.argmax(output_size, dim=1)
            correct_number += torch.sum(pred_number == target_list[0]).item()
            correct_size += torch.sum(pred_size == target_list[1]).item()

    test_accuracy_number = 100. * correct_number / len(test_loader.dataset)        
    test_accuracy_size = 100. * correct_size / len(test_loader.dataset)

    return test_accuracy_number, test_accuracy_size


def resize_transform(aux):
    return transforms.Resize(aux[None, None,:].shape[1:])[0,0]

def gen_gradcams(data_loader, parse):
    
    img_idx = 0

    for batch_idx, (data, target) in enumerate(test_loader):
        res = torch.zeros((data_loader.size, args.image_size, args.image_size))
        targets = torch.zeros((data_loader.size,))

        for ix in range(data.shape[0]):
          img = data[ix]
          # resize_transform = 
          aux = get_gradcam(model, img, target[ix])
          proyected_grad_cam = resize_transform(aux)
          res[img_idx] = proyected_grad_cam
          target[img_idx] = target[ix]

          img_idx += 1

          
# Load the model
model = LeNet([args.image_size, args.image_size, 1], [4, 2])
model.load_state_dict(torch.load(args.model_destination + 'model.pt'))
model.eval()
model = model.to(device)
number_accuracy, size_accuracy = test(model, test_loader)
print('Test Number blobs accuracy: {:.2f}%'.format(number_accuracy))
print('Test Size accuracy: {:.2f}%'.format(size_accuracy))



