import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import torch
import os
import json
from os.path import isdir
from torch import nn, optim
from torch.autograd import variable
import torch.nn as nn
import torch.functional as F
import torchvision
import argparse
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from collections import OrderedDict

#**************************************************************************************************
def CL_args():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, help='Path to dataset')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--arch', type=str, help='Model architecture')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, help='Number of hidden units')
    parser.add_argument('--checkpoint', type=str, help='Save trained model checkpoint to file')
    parser.add_argument('--save_dir', dest= 'save_dir', type=str, default='/home/workspace/ImageClassifier', action='store')
    

    args = parser.parse_args()
    return args
#-----------------------------------------------------------------------------------------------
def save_chkPt(Model, Save_Dir, Train_data):
       
    # Save model at checkpoint
    if type(Save_Dir) == type(None):
        print("Need Model checkpoint directory, model not saved.")
    else:
         if isdir(Save_Dir):
               # Create class_to_idx attribute in model
               Model.class_to_idx = Train_data.class_to_idx
               # Create checkpoint dictionary
               checkpoint = {'architecture': Model.name,
                             'classifier': Model.classifier,
                             'class_to_idx': Model.class_to_idx,
                             'state_dict': Model.state_dict()}
              # Save checkpoint
               if type(args.checkpoint) == type(None):
                  torch.save(checkpoint, "checkpoint.pth")
               else:
                  torch.save(checkpoint, f"{args.checkpoint}.pth")
            
               print('---------The model is saved---------')
         else: 
              print("Directory not found, model  not saved.")      
         
#**************************************************************************************************
args = CL_args()
# ----------------------------Training data processes-------------------------------
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

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
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data= datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data= datasets.ImageFolder(test_dir, transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
#------------------------------------------------------------------------------------


#------loading cat_to_name.json file------
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
#-----------------------------------------  

#----------Checking for the device--------
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Your machine support GPU\n")
else:
    device = torch.device("cpu")
    print("Your machine support only CPU\n")
#-----------------------------------------


#----------Using VGG16 model---------------------------------------
if type(args.arch) == type(None): 
      model = models.vgg13(pretrained=True)
      model.name = "vgg13"
      print("Network architecture is vgg13.")
else: 
      exec("model = models.{}(pretrained=True)".format(args.arch))
      model.name = args.arch
#--------------------------------------------------------------------


#---Freazing The parameters from the gradiant---
for param in model.parameters():
    param.requires_grad = False
#----------------------------------------------


#-----Setting new classifier for the model, criterion and optimizer---------------

#-----------setting learning rate----------
if type(args.hidden_units) == type(None): 
        hidden_units = 4096 #hyperparamters
        print("Number of Hidden Layers is 4096.")
else: 
    hidden_units = args.hidden_units
#-------------------------------------------

input_features = model.classifier[0].in_features        
        
model.classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_features, hidden_units)),
                                              ('relu1', nn.ReLU()),
                                              ('dropout1', nn.Dropout(p=0.2)),
                                              ('fc2', nn.Linear(hidden_units, round(hidden_units/3.5))),
                                              ('relu2', nn.ReLU()),
                                              ('dropout2', nn.Dropout(p=0.2)),
                                              ('fc3', nn.Linear(round(hidden_units/3.5) , 102)),
                                              ('output',  nn.LogSoftmax(dim=1))]))

criterion = nn.NLLLoss()

#-----------setting learning rate----------
if type(args.learning_rate) == type(None):
        learning_rate = 0.001
        print("Learning rate is 0.001")
else: 
     learning_rate = args.learning_rate
#------------------------------------------    

optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
model.to(device)
#-----------------------------------------------------------------------------------


#=========================Here our Training loop==================================
print("=========================The model is training Now=========================\n")

#-----------setting Epochs----------
if type(args.epochs) == type(None):
        epochs = 1
        print("Epochs = 1")
else: 
     print(f"Epochs = {args.epochs}")
     epochs = args.epochs
#-----------------------------------

steps = 0
running_loss = 0
print_every = 100

for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        #clear the gradiant
        optimizer.zero_grad()
        
        #------training process-------
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        #-----------------------------
        running_loss += loss.item()
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            
            model.eval()
            with torch.no_grad():
                
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    loss = criterion(logps, labels)
                    
                    test_loss += loss.item()
                    
                    #--------------------accuracy calculation------------------
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    #------------------------------------------------------------
                    
            print(f"Epoch {epoch+1}/{epochs}.. |"
                  f"Train loss: {running_loss/print_every:.3f}.. |"
                  f"Test loss: {test_loss/len(validloader):.3f}.. |"
                  f"Test accuracy: {100*(accuracy/len(validloader)):.2f}%")
            running_loss = 0
            model.train()
print("\n=====================The model has finished the training=====================\n")            
#=========================End of our Training loop================================


#===============================Validation loop==================================
if torch.cuda.is_available():
    model.cuda()
    
accuracy = 0
for inputs, labels in testloader:
    
    inputs, labels = inputs.to(device), labels.to(device)
    logps = model.forward(inputs)   
    
    #--------------------accuracy calculation------------------
    ps = torch.exp(logps)
    top_p, top_class = ps.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()   
    #------------------------------------------------------------
    
print(f"accuracy: {round(100 * (accuracy/len(testloader)))}%")
#=========================End of our Validation loop==============================

#===================The checkpoint of a model================
save_chkPt(model, args.save_dir, train_data)
#============================================================
#python train.py --epochs 7 --checkpoint raied_model    
#python train.py --checkpoint raied_model 