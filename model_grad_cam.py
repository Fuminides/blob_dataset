import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from PIL import Image


# Define the model
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 3)
        self.pool1 = nn.AvgPool2d(2)
        
        self.conv2 = nn.Conv2d(3, 3, 3)

        self.fc1 = nn.Linear(3 * 13 * 13, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        if self.training:
          h = x.register_hook(self.activations_hook)
        
        x = x.view(-1, 3 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)
   
    
    def get_activations_gradient(self):
        return self.gradients


    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
    
    # 
    def get_activations(self, x):
      x = self.conv1(x)
      x = self.pool1(x)
      x = self.conv2(x)
      return x


    def get_gradcam(self, image, target_class_index):
        # set the evaluation mode
        self.eval()

        # get the image from the dataloader
        img = image

        # get the most likely prediction of the model
        pred = self(img)

        # get the gradient of the output with respect to the parameters of the model
        pred[:, target_class_index].backward()

        # pull the gradients out of the model
        gradients = self.get_activations_gradient()

        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[1, 2])

        # get the activations of the last convolutional layer
        activations = self.get_activations(img).detach()

        # weight the channels by corresponding gradients
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
            
        # average the channels of the activations
        heatmap = torch.mean(activations, dim=0).squeeze()

        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        heatmap = np.maximum(heatmap, 0)

        # normalize the heatmap
        heatmap /= torch.max(heatmap)
        
        return heatmap
      
    
    def plot_grad_cam(self, image, target_class_index):
        heatmap = self.get_gradcam(image, target_class_index)

        return (heatmap* 0.7 +  image*0.3).permute(1, 2, 0)
    

    def get_saliency_map2(self, image, target_class_index):
        self.eval()
        og_shape = image.shape
        image = image.requires_grad_(True)
        output = self(image)
        loss = -output[0, target_class_index]
        self.zero_grad()
        loss.backward()
        saliency = image.grad
        
        return saliency[0]
        

model = Net()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# Train the model
def train(model, criterion, optimizer, train_loader, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


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

epochs = 20
for epoch in range(1, epochs + 1):
    print('Epoch ', epoch)
    train(model, criterion, optimizer, train_loader, epoch)