test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=1000, shuffle=True)


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

test_loss, test_accuracy = test(model, test_loader)
print('Epoch: {}/{} Test Loss: {:.4f} Test Accuracy: {:.2f}%'.format(
      epoch, epochs, test_loss, test_accuracy))

print('Training completed!')

import matplotlib.pyplot as plt
import torchvision.transforms as transforms



test_loader = torch.utils.data.DataLoader(dataset=cifar_testset, batch_size=64, shuffle=False)
ix = 0
for batch_idx, (data, target) in enumerate(test_loader):
  # saliency = get_saliency_map2(model, data[0], target[0])
  resize_transform = transforms.Resize(data[0].shape[1:])

  plt.figure()
  plt.imshow(data[0].permute(1, 2, 0))
  plt.figure()
  #plt.imshow((saliency > 0.0)* data[0, 0] )
  #plt.figure()
  aux = get_gradcam(model, data[0], target[0])
  aux = aux[None, None,:]
  plt.imshow(resize_transform(aux)[0,0])
  plt.figure()
  plt.imshow((resize_transform(aux)[0,0] * 0.7 +  data[0]*0.3).permute(1, 2, 0))
  
  plt.show()

  ix += 1

  if ix == 10:
    break