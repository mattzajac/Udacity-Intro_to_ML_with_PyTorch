import pandas as pd
import numpy as np
import os

import torch
from torch import nn, optim
from torch.optim import lr_scheduler

import torchvision
from torchvision import datasets, transforms, models

from PIL import Image
import json



with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


def process_and_load_data(data_dir = 'flowers'):
    
    train_dir = data_dir + '/train/'
    test_dir = data_dir + '/test/'
    valid_dir = data_dir + '/valid/'
    
    transform = {
        'train': transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]),
        'test': transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])]),
        'valid': transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    }
    
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform = transform['train']),
        'test': datasets.ImageFolder(test_dir, transform = transform['test']),
        'valid': datasets.ImageFolder(valid_dir, transform = transform['valid'])
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size = 64, shuffle = True),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size = 64, shuffle = True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size = 64, shuffle = True)
    }
    
    return image_datasets, dataloaders


def check_gpu(gpu_arg=True):
   # If gpu_arg is false then simply return the cpu device
    if gpu_arg:
        return torch.device("cpu")
    
    # If gpu_arg then make sure to check for CUDA before assigning it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return device


def load_chosen_model(arch):

    if arch == 'vgg19_bn':
        model = models.vgg19_bn(pretrained = True)
        input_size = model.classifier[0].in_features
    elif arch == 'densenet201':
        model = models.densenet201(pretrained = True)
        input_size = model.classifier.in_features
    elif arch == 'resnet152':
        model = models.resnet152(pretrained = True)
        input_size = model.fc.in_features
    return model, input_size


def build_model(arch = 'resnet152', hidden_units = 1024, learning_rate = 0.001):
    
    model, input_size = load_chosen_model(arch)
    
    # Turn-off the gradient
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(nn.Linear(input_size, hidden_units),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_units, 512),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(256, 102),
                        nn.LogSoftmax(dim=1))
    
    if 'vgg' in arch or 'densenet' in arch:
        model.classifier = classifier
    elif 'resnet' in arch:
        model.fc = classifier
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
        
    return model, optimizer, criterion


def train_and_validate_model(model, optimizer, criterion, dataloaders, device, epochs = 5, print_every = 32):

    model.to(device)
    steps = 0
    train_loss = 0

    for epoch in range(epochs):

        for data in iter(dataloaders['train']):
            inputs, labels = data
            steps += 1

            # Move input and label tensors to the default device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Clean existing gradients
            optimizer.zero_grad()

            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                valid_accuracy = 0

                with torch.no_grad():
                    for data in iter(dataloaders['valid']):
                        inputs, labels = data

                        # Move input and label tensors to the default device
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        logps = model(inputs)
                        loss = criterion(logps, labels)

                        valid_loss += loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_ps, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {train_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
                      f"Validation accuracy: {valid_accuracy/len(dataloaders['valid']):.3f}")

                train_loss = 0
                model.train()
                
    return model


def test_model(model, optimizer, criterion, dataloaders, device):

    model.eval()
    test_loss = 0
    test_accuracy = 0
            
    with torch.no_grad():
        for data in iter(dataloaders['test']):
            inputs, labels = data
                         
            # Move input and label tensors to the default device
            inputs = inputs.to(device)
            labels = labels.to(device)

            logps = model(inputs)
            loss = criterion(logps, labels)

            test_loss += loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_ps, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test loss: {test_loss/len(dataloaders['test']):.3f}.. "
          f"Test accuracy: {test_accuracy/len(dataloaders['test']):.3f}")
    

    
def save_checkpoint(arch, model, epochs, hidden_units, learning_rate, image_datasets, checkpoint_dir = 'checkpoint'):

    if checkpoint_dir:
        has_save_dir = os.path.exists(checkpoint_dir);
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        save_path = checkpoint_dir + '/' + arch + '_checkpoint.pth'
    else:
        save_path = arch + '_checkpoint.pth'
                

    # Create `class_to_idx` attribute in model
    model.class_to_idx = image_datasets['train'].class_to_idx

    # Create checkpoint dictionary
    checkpoint = {
        'architecture': arch,
        'epochs': epochs,
        'hidden_units': hidden_units,
        'learning_rate': learning_rate,
        'state_dict':model.state_dict(),
        'mapping':model.class_to_idx
    }

    # Save checkpoint
    torch.save(checkpoint, save_path)

    
def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath)
    
    arch = str(checkpoint['architecture'])
    hu = int(checkpoint['hidden_units'])
    lr = float(checkpoint['learning_rate'])
        
    model, optimizer, criterion = build_model(arch, hidden_units = hu, learning_rate = lr)
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['mapping']
     
    return model


def process_image(image_path):
    
    img = Image.open(image_path)
    adjust = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    img_example = adjust(img)
    
    return img_example


def predict(model, image_path, topk=5):

    model.to('cpu')
    
    processed_image = process_image(image_path)
    processed_image.unsqueeze_(0)
    probs = torch.exp(model.forward(processed_image))
    top_probs, top_labs = probs.topk(topk)

    idx_to_class = {}
    for key, value in model.class_to_idx.items():
        idx_to_class[value] = key

    np_top_labs = top_labs[0].numpy()

    top_labels = []
    for label in np_top_labs:
        top_labels.append(int(idx_to_class[label]))

    top_flowers = [cat_to_name[str(lab)] for lab in top_labels]
    
    return top_probs, top_labels, top_flowers


def print_probability(top_flowers, top_probs):
    
    top_probs = top_probs[0].detach().numpy()
    
    for i, j in enumerate(zip(top_flowers, top_probs)):
        print ("Rank {}:".format(i+1),
               "Flower: {}, probability: {}%".format(j[0], np.around(j[1]*100, decimals=2)))