import matplotlib.pyplot as plt
import torchvision.transforms as transforms

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
