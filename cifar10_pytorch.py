# coding: utf-8


import sys
from python_environment_check import check_packages
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision 
from torchvision import transforms 
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchsummary import summary
import time

# # Machine Learning with PyTorch and Scikit-Learn  
# # -- Code Examples

# ## Package version checks

# Add folder to path in order to load from the check_packages.py script:



sys.path.insert(0, '..')


# Check recommended package versions:





d = {
    'numpy': '1.21.2',
    'scipy': '1.7.0',
    'matplotlib': '3.4.3',
    'torch': '1.8.0',
    'torchvision': '0.9.0'
}
check_packages(d)


# # Chapter 14: Classifying Images with Deep Convolutional Neural Networks (Part 2/2)

# **Outline**
# 
# - [Smile classification from face images using a CNN](#Constructing-a-CNN-in-PyTorch)
#   - [Loading the CelebA dataset](#Loading-the-CelebA-dataset)
#   - [Image transformation and data augmentation](#Image-transformation-and-data-augmentation)
#   - [Training a CNN smile classifier](#Training-a-CNN-smile-classifier)
# - [Summary](#Summary)

# Note that the optional watermark extension is a small IPython notebook plugin that I developed to make the code reproducible. You can just skip the following line(s).







# ## Smile classification from face images using CNN
# 

# ### Loading the CelebA dataset

# You can try setting `download=True` in the code cell below, however due to the daily download limits of the CelebA dataset, this will probably result in an error. Alternatively, we recommend trying the following:
# 
# - You can download the files from the official CelebA website manually (https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 
# - or use our download link, https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing (recommended). 
# 
# If you use our download link, it will download a `celeba.zip` file, 
# 
# 1. which you need to unpack in the current directory where you are running the code. 
# 2. In addition, **please also make sure you unzip the `img_align_celeba.zip` file, which is inside the `celeba` folder.**
# 3. Also, after downloading and unzipping the celeba folder, you need to run with the setting `download=False` instead of `download=True` (as shown in the code cell below).
# 
# In case you are encountering problems with this approach, please do not hesitate to open a new issue or start a discussion at https://github.com/ rasbt/machine-learning-book so that we can provide you with additional information.


SHOW = False

image_path = './'
cifar10_train_dataset = torchvision.datasets.CIFAR10(image_path, train=True, download=True)
cifar10_test_dataset = torchvision.datasets.CIFAR10(image_path, train=False, download=True)

train_ds_shape = cifar10_train_dataset.data.shape
test_ds_shape = cifar10_test_dataset.data.shape
train_len = train_ds_shape[0]
img_height = train_ds_shape[1]
img_width = train_ds_shape[2]
channels = train_ds_shape[3]

print('Train set shape:', train_ds_shape)
print('Test set shape:', test_ds_shape)
print(f'img height: {img_height} img width: {img_width} number of channels: {channels}')


# ### Image transformation and data augmentation




## take 5 examples

fig = plt.figure(figsize=(16, 8.5))

## Column 1: cropping to a bounding-box
ax = fig.add_subplot(2, 5, 1)
img, attr = cifar10_train_dataset[0]
ax.set_title('Crop to a \nbounding-box', size=15)
ax.imshow(img)
ax = fig.add_subplot(2, 5, 6)
img_cropped = transforms.functional.crop(img, 2, 2, 28, 28)
ax.imshow(img_cropped)

## Column 2: flipping (horizontally)
ax = fig.add_subplot(2, 5, 2)
img, attr = cifar10_train_dataset[1]
ax.set_title('Flip (horizontal)', size=15)
ax.imshow(img)
ax = fig.add_subplot(2, 5, 7)
img_flipped = transforms.functional.hflip(img)
ax.imshow(img_flipped)

## Column 3: adjust contrast
ax = fig.add_subplot(2, 5, 3)
img, attr = cifar10_train_dataset[2]
ax.set_title('Adjust constrast', size=15)
ax.imshow(img)
ax = fig.add_subplot(2, 5, 8)
img_adj_contrast = transforms.functional.adjust_contrast(img, contrast_factor=2)
ax.imshow(img_adj_contrast)

## Column 4: adjust brightness
ax = fig.add_subplot(2, 5, 4)
img, attr = cifar10_train_dataset[3]
ax.set_title('Adjust brightness', size=15)
ax.imshow(img)
ax = fig.add_subplot(2, 5, 9)
img_adj_brightness = transforms.functional.adjust_brightness(img, brightness_factor=1.3)
ax.imshow(img_adj_brightness)

## Column 5: cropping from image center 
ax = fig.add_subplot(2, 5, 5)
img, attr = cifar10_train_dataset[4]
ax.set_title('Center crop\nand resize', size=15)
ax.imshow(img)
ax = fig.add_subplot(2, 5, 10)
img_center_crop = transforms.functional.center_crop(img, [28, 28])
img_resized = transforms.functional.resize(img_center_crop, size=(32, 32))
ax.imshow(img_resized)
 
# plt.savefig('figures/14_14.png', dpi=300)
if SHOW:
    plt.show()




torch.manual_seed(1)

fig = plt.figure(figsize=(14, 12))

for i, (img, attr) in enumerate(cifar10_train_dataset):
    ax = fig.add_subplot(3, 4, i*4+1)
    ax.imshow(img)
    if i == 0:
        ax.set_title('Orig.', size=15)
        
    ax = fig.add_subplot(3, 4, i*4+2)
    img_transform = transforms.Compose([transforms.RandomCrop([26, 26])])
    img_cropped = img_transform(img)
    ax.imshow(img_cropped)
    if i == 0:
        ax.set_title('Step 1: Random crop', size=15)

    ax = fig.add_subplot(3, 4, i*4+3)
    img_transform = transforms.Compose([transforms.RandomHorizontalFlip()])
    img_flip = img_transform(img_cropped)
    ax.imshow(img_flip)
    if i == 0:
        ax.set_title('Step 2: Random flip', size=15)

    ax = fig.add_subplot(3, 4, i*4+4)
    img_resized = transforms.functional.resize(img_flip, size=(24, 28))
    ax.imshow(img_resized)
    if i == 0:
        ax.set_title('Step 3: Resize', size=15)
    
    if i == 2:
        break
        
# plt.savefig('figures/14_15.png', dpi=300)
if SHOW:
    plt.show()




get_smile = lambda attr: attr[18]
 
transform_train = transforms.Compose([
    transforms.RandomCrop([30, 30]),
    transforms.RandomHorizontalFlip(),
    transforms.Resize([32, 32]),
    transforms.ToTensor(),
])

transform = transforms.Compose([
    transforms.CenterCrop([30, 30]),
    transforms.Resize([32, 32]),
    transforms.ToTensor(),
])



cifar10_train_dataset = torchvision.datasets.CIFAR10(image_path, train=True, download=True, transform=transform_train)

torch.manual_seed(1)
data_loader = DataLoader(cifar10_train_dataset, batch_size=2)

fig = plt.figure(figsize=(15, 6))

num_epochs = 5
for j in range(num_epochs):
    img_batch, label_batch = next(iter(data_loader))
    img = img_batch[0]
    ax = fig.add_subplot(2, 5, j + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Epoch {j}:', size=15)
    ax.imshow(img.permute(1, 2, 0))

    img = img_batch[1]
    ax = fig.add_subplot(2, 5, j + 6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(img.permute(1, 2, 0))
      
    
#plt.savefig('figures/14_16.png', dpi=300)
if SHOW:
    plt.show()


cifar10_train_dataset = torchvision.datasets.CIFAR10(image_path, train=True, download=True, transform=transform)

original_cifar10_train_dataset = cifar10_train_dataset
cifar10_train_dataset = Subset(original_cifar10_train_dataset, torch.arange(0, 45000, 1)) 
cifar10_valid_dataset = Subset(original_cifar10_train_dataset, torch.arange(45001, train_len-1, 1)) 
 
print('Train set:', len(cifar10_train_dataset))
print('Validation set:', len(cifar10_valid_dataset))




batch_size = 32

torch.manual_seed(1)
train_dl = DataLoader(cifar10_train_dataset, batch_size, shuffle=True)
valid_dl = DataLoader(cifar10_valid_dataset, batch_size, shuffle=False)
test_dl = DataLoader(cifar10_test_dataset, batch_size, shuffle=False)


# ### Training a CNN Smile classifier
# 
# * **Global Average Pooling**








model = nn.Sequential()

model.add_module('conv1', nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1))
model.add_module('relu1', nn.ReLU())        
model.add_module('pool1', nn.MaxPool2d(kernel_size=2))  
model.add_module('dropout1', nn.Dropout(p=0.1)) 

# model.add_module('conv1_2', nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1))
# model.add_module('relu1_2', nn.ReLU())        
# model.add_module('pool1_2', nn.MaxPool2d(kernel_size=2))  
# model.add_module('dropout1_2', nn.Dropout(p=0.1)) 

model.add_module('conv2', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1))
model.add_module('relu2', nn.ReLU())        
model.add_module('pool2', nn.MaxPool2d(kernel_size=2))   
model.add_module('dropout2', nn.Dropout(p=0.1)) 

# model.add_module('conv2_2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1))
# model.add_module('relu2_2', nn.ReLU())        
# model.add_module('pool2_2', nn.MaxPool2d(kernel_size=2))   
# model.add_module('dropout2_2', nn.Dropout(p=0.1)) 

model.add_module('conv3', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1))
model.add_module('relu3', nn.ReLU())        
model.add_module('pool3', nn.MaxPool2d(kernel_size=2))   

# model.add_module('conv3_2', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1))
# model.add_module('relu3_2', nn.ReLU())        
# model.add_module('pool3_2', nn.MaxPool2d(kernel_size=2)) 

model.add_module('conv4', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1))
model.add_module('relu4', nn.ReLU())  

# model.add_module('conv4_2', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1))
# model.add_module('relu4_2', nn.ReLU())  




x = torch.ones((4, 3, 32, 32))
model(x).shape




model.add_module('pool4', nn.AvgPool2d(kernel_size=4)) 
model.add_module('flatten', nn.Flatten()) 

x = torch.ones((4, 3, 32, 32))
model(x).shape




model.add_module('fc', nn.Linear(512, 10)) 
model.add_module('sigmoid', nn.Sigmoid()) 




x = torch.ones((4, 3, 32, 32))
model(x).shape




model

# summary(model, input_size = (3, 32, 32), batch_size = -1)
summary(model.cuda(), input_size = (3, 32, 32))




device = torch.device("cuda:0")
# device = torch.device("cpu")
model = model.to(device) 




loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(model, num_epochs, train_dl, valid_dl):
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        for x_batch, y_batch in train_dl:
            x_batch = x_batch.to(device) 
            y_batch = y_batch.to(device) 
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist_train[epoch] += loss.item()*y_batch.size(0)
            pred_max = pred.argmax(axis=1)
            is_correct = (pred_max.float() == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.sum().cpu()

        loss_hist_train[epoch] /= len(train_dl.dataset)
        accuracy_hist_train[epoch] /= len(train_dl.dataset)
        
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                x_batch = x_batch.to(device) 
                y_batch = y_batch.to(device) 
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss_hist_valid[epoch] += loss.item()*y_batch.size(0) 
                pred_max = pred.argmax(axis=1)
                is_correct = (pred_max.float() == y_batch).float()
                accuracy_hist_valid[epoch] += is_correct.sum().cpu()

        loss_hist_valid[epoch] /= len(valid_dl.dataset)
        accuracy_hist_valid[epoch] /= len(valid_dl.dataset)
        end_time = time.time()
        print(f'Epoch {epoch+1} accuracy: {accuracy_hist_train[epoch]:.4f} val_accuracy: {accuracy_hist_valid[epoch]:.4f} elapsed time: {end_time-start_time}')
    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid

torch.manual_seed(1)
num_epochs = 30
hist = train(model, num_epochs, train_dl, valid_dl)




x_arr = np.arange(len(hist[0])) + 1

fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 2, 1)
ax.plot(x_arr, hist[0], '-o', label='Train loss')
ax.plot(x_arr, hist[1], '--<', label='Validation loss')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Loss', size=15)

ax = fig.add_subplot(1, 2, 2)
ax.plot(x_arr, hist[2], '-o', label='Train acc.')
ax.plot(x_arr, hist[3], '--<', label='Validation acc.')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Accuracy', size=15)

#plt.savefig('figures/14_17.png', dpi=300)
plt.show()




accuracy_test = 0

model.eval()
with torch.no_grad():
    for x_batch, y_batch in test_dl:
        x_batch = x_batch.to(device) 
        y_batch = y_batch.to(device) 
        pred = model(x_batch)[:, 0]
        is_correct = ((pred>=0.5).float() == y_batch).float()
        accuracy_test += is_correct.sum().cpu()
 
accuracy_test /= len(test_dl.dataset)
        
print(f'Test accuracy: {accuracy_test:.4f}') 





path = 'models/cifar10-cnn.ph'
torch.save(model, path)


# ...
# 
# 
# ## Summary
# 
# ...
# 
# 

# ----
# 
# Readers may ignore the next cell.









