# Imports here
import matplotlib.pyplot as plt

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('data_dir', action="store")
parser.add_argument('--save_dir', action="store", dest="save_dir", default='checkpoint')
parser.add_argument('--arch', action="store", dest="arch", default='vgg16')
parser.add_argument('--learning_rate', action="store", dest="learning_rate", default=0.001)
parser.add_argument('--hidden_units', action="store", dest="hidden_units", default=500)
parser.add_argument('--epochs', action="store", dest="epochs", default=3)
parser.add_argument('--gpu', action="store_const", dest="device", const="gpu", default='cpu')

results = parser.parse_args()
data_dir = results.data_dir
save_dir = results.save_dir
arch = results.arch
learning_rate = results.learning_rate
hidden_units = results.hidden_units
epochs = results.epochs
device = results.device

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
    
# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


# TODO: Load the datasets with ImageFolder
train_image_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_image_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_image_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size=64, shuffle=True)
valid_dataloaders = torch.utils.data.DataLoader(valid_image_datasets, batch_size=64)
test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size=64)

# TODO: Build and train your network
if arch == 'vgg13':
    model = models.vgg13(pretrained=True)
else:
    model = models.vgg16(pretrained=True)

for param in model.parameters():
    param.requires_grad = False
    
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(25088, hidden_units)),
    ('relu', nn.ReLU()),
    ('dropout', nn.Dropout(0.5)),
    ('fc2', nn.Linear(hidden_units,102)),
    ('output', nn.LogSoftmax(dim=1))
]))

model.classifier = classifier
    
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

# Implement a function for the validation pass
def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:
        
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

def do_deep_learning(model, trainloader, testloader, epochs, print_every, criterion, optimizer, device='cpu'):
    epochs = epochs
    print_every = print_every
    steps = 0

    # change to cuda
    if device == 'gpu':
        device = 'cuda'
    model.to(device)

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()
            
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation(model, testloader, criterion,device)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
                
                running_loss = 0
                
                # Make sure training is back on
                model.train()

do_deep_learning(model, train_dataloaders, valid_dataloaders, epochs, 40, criterion, optimizer, device)

# TODO: Do validation on the test set

def check_accuracy_on_test(testloader, device):    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            if device == 'gpu':
                device = 'cuda'
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

check_accuracy_on_test(test_dataloaders, device)

    # TODO: Save the checkpoint 
print("Our model: \n\n", model, '\n')
print("The state dict keys: \n\n", model.state_dict().keys())
model.class_to_idx = train_image_datasets.class_to_idx
model.cpu()
torch.save({'arch':arch,'state_dict':model.state_dict(),'class_to_idx':model.class_to_idx},save_dir+'.pth')