import torch
import torch.nn as nn
from model_grad_cam import LeNet
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from parser import gen_parser
from blob_dataloader import BlobDataset

# Parse the arguments
parser = gen_parser()
args = parser.parse_args()

# Load the data
if args.dataset == 'mnist':
    train_dataset = datasets.MNIST(root='.', train=True, transform=transforms.ToTensor(), download=True)
elif args.dataset == 'cifar10':
    train_dataset = datasets.CIFAR10('data', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))
                         ]))
elif args.dataset == 'blob':
    dataset = BlobDataset('trials/', train=True)
    

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

# Define the model
model = LeNet([args.image_size, args.image_size, 3], 512)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# Train the model
def train(model, criterion, optimizer, train_loader, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()




epochs = args.epochs
for epoch in range(1, epochs + 1):
    print('Epoch ', epoch)
    train(model, criterion, optimizer, train_loader, epoch)

# Save the model
torch.save(model.state_dict(), args.model_destination + 'model.pt')