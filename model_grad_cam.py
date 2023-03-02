import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image

import torch
import torch.nn as nn

class LeNet(nn.Module):

    def __init__(self, input_size, output_size, num_filters=[64, 64],
                 fc_sizes=[384, 192, -1], use_batch_norm=True, dropout=0.0):
        super().__init__()
        
        self.relu = nn.ReLU()
        self.use_batch_norm = use_batch_norm

        # Layer "group" 0:
        self.conv_0 = nn.Conv2d(in_channels=input_size[2], out_channels=num_filters[0],
                                   kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.pool_0 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        if use_batch_norm:
            self.batch_norm_0 = nn.BatchNorm2d(num_filters[0])
        # Layer group 1:
        self.conv_1 = nn.Conv2d(num_filters[0], out_channels=num_filters[1],
                                kernel_size=(3, 3), padding=(1, 1), bias=True)
        self.pool_1 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        if use_batch_norm:
            self.batch_norm_1 = nn.BatchNorm2d(num_filters[1])
        self.dropout2d = nn.Dropout2d(p=dropout)
        # MLP Classifier:
        self.flatten = nn.Flatten()
        flattened_size = int((input_size[0] / (2 * 2)) * (input_size[1] / (2 * 2))) * num_filters[1]
        self.fc_0 = nn.Linear(flattened_size, fc_sizes[0], bias=True)
        self.fc_1 = nn.Linear(fc_sizes[0], fc_sizes[1], bias=True)
        self.fc_2 = nn.Linear(fc_sizes[1], output_size, bias=True)
        self.dropout1d = nn.Dropout(p=dropout)

    def forward(self, x):
        output_block_0 = self.pool_0(self.relu(self.conv_0(x)))
        output_block_0 = self.dropout2d(output_block_0)
        if self.use_batch_norm:
             output_block_0 = self.batch_norm_0(output_block_0)
        output_block_1 = self.pool_1(self.relu(self.conv_1(output_block_0)))
        output_block_1 = self.dropout2d(output_block_1)
        if self.use_batch_norm:
            output_block_1 = self.batch_norm_1(output_block_1)

        if self.training:
          h = output_block_1.register_hook(self.activations_hook)

        output_flattened = self.flatten(output_block_1)
        output_fc_0 = self.relu(self.fc_0(output_flattened))
        output_fc_0 = self.dropout1d(output_fc_0)
        output_fc_1 = self.relu(self.fc_1(output_fc_0))
        output_fc_1 = self.dropout1d(output_fc_1)
        predictions = self.fc_2(output_fc_1)  # Softmax will be applied to this value
        return predictions
      
    def get_activations_gradient(self):
        return self.gradients

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
      
    def get_activations(self, x):
      output_block_0 = self.pool_0(self.relu(self.conv_0(x)))
      output_block_0 = self.dropout2d(output_block_0)
      if self.use_batch_norm:
            output_block_0 = self.batch_norm_0(output_block_0)
      output_block_1 = self.pool_1(self.relu(self.conv_1(output_block_0)))
      output_block_1 = self.dropout2d(output_block_1)
      if self.use_batch_norm:
          output_block_1 = self.batch_norm_1(output_block_1)
      
      return output_block_1

# Define the model
class BasicNet(nn.Module):

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
        

def get_gradcam(model, image, target_class_index):
        # set the evaluation mode
        model.eval()

        # get the image from the dataloader
        img = image

        # get the most likely prediction of the model
        pred = model(img)

        # get the gradient of the output with respect to the parameters of the model
        pred[:, target_class_index].backward()

        # pull the gradients out of the model
        gradients = model.get_activations_gradient()

        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[1, 2])

        # get the activations of the last convolutional layer
        activations = model.get_activations(img).detach()

        # weight the channels by corresponding gradients
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
            
        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze().cpu()

        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        heatmap = np.maximum(heatmap, 0)

        # normalize the heatmap
        heatmap /= torch.max(heatmap)
        
        return heatmap

class MyResNet(nn.Module):

    def __init__(self, num_class, end_dim=128):
        super(MyResNet, self).__init__()

        # Load pre-trained visual model
        resnet = models.resnet50(pretrained=True)
        self.visual_resnet = nn.Sequential(*list(resnet.children())[0:5])
        self.avg_pooling_resnet = nn.Sequential(*list(resnet.children())[5:-2])

        self.deep_feature_size = 512
        # Classifier
        self.classifier1 = nn.Sequential(nn.Linear(2048, self.deep_feature_size))
        self.classifier2 = nn.Sequential(nn.Linear(self.deep_feature_size, num_class))

        # Graph space encoder
        self.nodeEmb = nn.Sequential(nn.Linear(2048, end_dim))
      
    def forward(self, img):
        resnet_emb = self.visual_resnet(img)
        if self.training:
          h = resnet_emb.register_hook(self.activations_hook)

        resnet_emb = self.avg_pooling_resnet(resnet_emb)
        resnet_emb1 = resnet_emb.view(resnet_emb.size(0), -1)
        pred_class = self.classifier1(resnet_emb1)
        pred_class = self.classifier2(pred_class)
        graph_proj = self.nodeEmb(resnet_emb1)

        return pred_class
      
    def get_activations_gradient(self):
        return self.gradients

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
      
    def get_activations(self, x):
      visual_emb = self.visual_resnet(x)
      
      return visual_emb


