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

    parser = argparse.ArgumentParser("Predict network settings")
    parser.add_argument('--image', default='flowers/test/3/image_06641.jpg', nargs='*', action="store", type = str) 
    parser.add_argument('--checkpoint', default = '/home/workspace/ImageClassifier/', nargs='*', action = 'store', type= str)
    parser.add_argument('--topk', default='3', dest ='topk', action='store', type =int,   help='TopK matches as int')
    parser.add_argument('--category_names', dest='category_names',  action= 'store', default= '/home/workspace/ImageClassifier/cat_to_name.json',
                        help='mapping categories to names')
    parser.add_argument('--gpu',  default = 'gpu', action='store', dest='gpu', help='Use GPU + Cuda calculations')
    

    args = parser.parse_args()
    return args
#-----------------------------------------------------------------------------------------------
def chkPt_loader(path):
    
    checkpoint = torch.load(path)
    
 
    if checkpoint['architecture'] == 'vgg13':
        model = models.vgg13(pretrained=True)
        model.name = "vgg13"
    else: 
        exec("model = models.{}(pretrained=True)".checkpoint['architecture'])
        model.name = checkpoint['architecture']
        
    for param in model.parameters(): 
        param.requires_grad = False    
    
    model.classifier = checkpoint['classifier']  
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx'] 
    
    return model , checkpoint
#-----------------------------------------------------------------------------------------------

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image).convert("RGB")
    pil_trans = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    transformed_Image = pil_trans(pil_image)
    return transformed_Image

#-----------------------------------------------------------------------------------------------
def predict(image, model, topk, modelCTI):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
   # TODO: Implement the code to predict the class from an image file
    model.eval();
    model.to('cpu')

    image = image.unsqueeze(0)
    
    with torch.no_grad():
        logps = model.forward(image)
        ps = torch.exp(logps)
        top_probs, top_labels = torch.topk(ps, k = topk)
        
    np_top_labels = top_labels.numpy()[0]
    np_top_probs = top_probs.numpy()[0]
    
    class_to_idx_inv = {modelCTI[k]: k for k in modelCTI}
    top_classes = []
    
    for label in np_top_labels:
        top_classes.append(class_to_idx_inv[label])
        
    top_probs = np_top_probs.tolist()     
    return top_probs, top_classes
#-----------------------------------------------------------------------------------------------
def ps_printing(top_ps, top_classes, cat_to_name):
    
    image_path = args.image
    image_split = image_path.split('/')
    target_class = cat_to_name['{}'.format(image_split[2])]
    
    for label, probs in zip(top_classes, top_ps):
        print ("Flower: {}, Percent Chance: {}%".format(cat_to_name[label], round(probs*100)))
    
    
    


#===============================================================================================
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

                                     


train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data= datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data= datasets.ImageFolder(test_dir, transform=test_transforms)


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

#----Loading The Model and Model Checkpoint----
Chk = args.checkpoint
model , ChkPt = chkPt_loader(Chk[0])
#----------------------------------------------

model.class_to_idx = train_data.class_to_idx
model.to(device)
#=================Testing the model accuracy==========================
print('===============Testing the model accuracy===============')
accuracy = 0
for inputs, labels in testloader:
    
    inputs, labels = inputs.to(device), labels.to(device)
    logps = model.forward(inputs)   
    # for accuracy calculation
    ps = torch.exp(logps)
    top_p, top_class = ps.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()    
    
print(f"accuracy: {round(100 * (accuracy/len(testloader)))}%")
#=====================================================================
#python predict.py --checkpoint raied_model.pth  

#====================Processing The Image==============================
transed_image = process_image(args.image)
#======================================================================


#================================model prediction=================================
top_ps, top_classes = predict(transed_image, model, args.topk, model.class_to_idx)
#=================================================================================

ps_printing(top_ps, top_classes, cat_to_name)