# Curious to see how, in the end, we need some manual work in order to get automate work

from skimage import filters
from torchsummary import summary
import torch
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

PATH = PATH

images = []

for directory, _, files in os.walk(PATH):
    for file in files:
        images.append(directory+'/'+file)

images = [i for i in images if '.jpg' in i or '.png' in i]

chunked_images = images[:100]

pics = []

for i in chunked_images:
    image = Image.open(i)
    image = image.resize((500, 500))
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    pic = np.array(image)
    image.close()

    pics.append(pic)

pics = np.array(pics)
pics = np.stack(pics, 0)
        
for image in range(pics.shape[0]):

    fig, ax = plt.subplots(3,3)
    for x in range(ax.shape[0]):
        for y in range(ax.shape[1]):
            ax[x,y].axis("off")
            
    img = Image.fromarray(pics[image])
    img = np.array(img)
    ax[0,1].imshow(img)
    ax[0,1].set_title("Original RGB image")
    
    Rthresh = filters.threshold_triangle(img[:, :, 0])
    R = img[:, :, 0] > Rthresh
    R = np.ones(R.shape) * R
    ax[2,0].imshow(R, cmap='gray')
    ax[2,0].set_title("Red Threshold")
    
    Gthresh = filters.threshold_triangle(img[:, :, 1])
    G = img[:, :, 1] > Gthresh
    G = np.ones(G.shape) * G
    ax[2,1].imshow(G, cmap='gray')
    ax[2,1].set_title("Green Threshold")
    
    Bthresh = filters.threshold_triangle(img[:, :, 2])
    B = img[:, :, 0] > Bthresh
    B = np.ones(B.shape) * B
    ax[2,2].imshow(B, cmap='gray')
    ax[2,2].set_title("Blue Threshold")
    
    plt.show()

    
best_channel = [0, 1, 2]

def create_mask(data, channels):
    masks = []
    for i in range(data.shape[0]):
        mask = filters.threshold_triangle(data[i, :, :, channels[i]])
        mask = data[i, :, :, channels[i]] > mask
        mask = np.ones(mask.shape) * mask

        masks.append(mask)
    
    masks = np.array(masks)
    masks = np.stack(masks, 0)
    
    return masks
  
