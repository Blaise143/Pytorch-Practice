import torch, torchvision
import pandas as pd


# loading in a pretrained model
model = torchvision.models.resnet18(pretrained= True)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

'''Making a forward pass'''
predicton = model(data)

'''Calculating loss and doing backward propagation'''
loss = (predicton - labels).sum()

loss.backward() # Backward pass

'''OPTIMIZATION STEP'''
# Loading an optimizer
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# initiating Gradient descent
optim.step()