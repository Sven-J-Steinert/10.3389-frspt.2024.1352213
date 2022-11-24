# imports

import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from PIL import Image, ImageTk, ImageMath, ImageOps
Image.MAX_IMAGE_PIXELS = 1000000000
import PIL.ImageGrab as ImageGrab

import sys

import xarray as xr

from os import environ
environ["OPENCV_IO_ENABLE_JASPER"] = "true"
import cv2

import urllib
import os

import geopandas
import shapely
shapely.geos.geos_version
from shapely.ops import cascaded_union

# helper functions

def orderOfMagnitude(number):
    return math.floor(math.log(abs(number), 10))

def resize_map(path):
    print("resize",path, 'to', resolution_im)
    image = Image.open("maps/" + path)
    print("original map image", "with extrema: ",image.getextrema())
    new_image = image.resize(resolution_im)
    print("resized map image", "with extrema: ",new_image.getextrema())

    return new_image

def resize_map_keep_min(path):
    print("resize",path, 'to', resolution_im)
    image = Image.open("maps/" + path)
    print("original map image", "with extrema: ",image.getextrema())
    min = image.getextrema()[0]
    new_image = image.resize(resolution_im)
    new_image.save('temp.png')
    new_image = cv2.imread('temp.png',0)
    new_image[new_image<min] = min # set everything below minimum to original minimum (1% to 1 w% = 22)
    cv2.imwrite('temp.png', new_image)
    new_image = Image.open("temp.png")
    print("resized map image", "with extrema: ",new_image.getextrema())
    os.remove("temp.png")

    return new_image

def read_im_values(im,value_divider):
    data = im.load()
    im_res = im.size

    Lat_use = np.linspace(Lat_max,Lat_min,resolution[1])
    print('Lat_use',np.min(Lat_use),np.max(Lat_use),Lat_use.shape, 'first Lat', Lat_use[0:5])
    Lon_use = np.linspace(Lon_min,Lon_max,resolution[0])
    print('Lon_use',np.min(Lon_use),np.max(Lon_use),Lon_use.shape, 'first Lon', Lon_use[0:5])

    print("     reading image values", "with extrema: ",im.getextrema())

    x = np.zeros(shape=(tuple((resolution[1],resolution[0]))))

    for n in tqdm(range(resolution[1])):

        index_y = round(np.interp(Lat_use[n], (-90,90), (im_res[1]-1,0)))

        for i in range(resolution[0]):
            index_x = round(np.interp(Lon_use[i], (-180, 180), (0, im_res[0]-1)))
            x[n][i] = data[index_x,index_y]/value_divider

    return x

        
def plot_map(values,value_devider,value_label,Lat_range,Lon_range,labelsize=None,save=None,bw=False,dpi=200,mass=False):
    
        if not labelsize: labelsize = 20
    
        Lat_min, Lat_max = Lat_range
        Lon_min, Lon_max = Lon_range
        
        min_value = np.min(values)
        max_value = np.max(values)
        
        print("display values",values.shape[::-1])
        print("extrema",min_value,max_value)
        
        plt.figure(figsize=(12,6), dpi=dpi)
        
        ax = plt.gca()
        if bw: im = ax.imshow(values, cmap='gray', interpolation='None', extent=[Lon_min,Lon_max,Lat_min,Lat_max])
        else: im = ax.imshow(values, cmap='viridis', interpolation='None', extent=[Lon_min,Lon_max,Lat_min,Lat_max])
            
        plt.xticks(np.arange(Lon_min, Lon_max+1, Lon_max/4))
        plt.yticks(np.arange(Lat_min, Lat_max+1, Lat_max/2))
        ax.tick_params(axis='both', which='major', labelsize=labelsize)
        
        if not bw:
            # create an axes on the right side of ax. The width of cax will be 2%
            # of ax and the padding between cax and ax will be fixed at 0.4 inch.
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size=0.25, pad=0.4)
            cbar = plt.colorbar(im, cax=cax)
            cbar.ax.set_ylabel(value_label, rotation=90, size=labelsize)

            #labels = np.append(np.arange(min_value, max_value, value_devider), max_value)

            # to have suitable ticks on cbar: find range from  [min .. max]
            # devide it into steps 1 order lower than difference results in ~10 ticks
            delta = (max_value/value_devider) - (min_value/value_devider)
            print('orderOfMagnitude(delta)',orderOfMagnitude(delta))
            maximum_flat_tick = 10**orderOfMagnitude(delta)
            one_step = (maximum_flat_tick/10) * value_devider
            min_flat_value = 10**orderOfMagnitude(min_value/value_devider)*value_devider
            
            if not mass:
                print('inside normal bottom scaling')
                labels = np.append( np.arange(min_value, max_value, one_step), max_value)
            else:
                print('inside weird bottom scaling')
                while min_flat_value - (min_value + one_step) < 0 : min_flat_value += one_step
                print('max_value',max_value)
                labels = np.append(np.append(min_value, np.arange(min_flat_value, max_value, one_step )), max_value)
                print(labels)
                

            print('image value spread',labels)
            loc    = labels

            cbar.set_ticks(loc)

            if labels[0] == round(labels[0]):
                cbar.set_ticklabels(['{:.0f}'.format(x) for x in labels])
            else:
                cbar.set_ticklabels(labels)

            cbar.ax.tick_params(labelsize=labelsize)
            plt.tight_layout()
            plt.draw()

            labels = [item for item in cbar.ax.get_yticks()]

            converted_value_labels = []

            for i in labels:
                converted_value_labels.append(i/value_devider)

            #converted_value_labels = np.append(np.arange(min_value/value_devider, max_value/value_devider, 1), max_value/value_devider)
            labels = converted_value_labels

            formatted_labels = []
            if mass:
                for i , x in enumerate(labels):
                    if i == 0 : formatted_labels.append('{:.2f}'.format(x/10**3))
                    elif i == len(labels)-1: formatted_labels.append('{:.2f}'.format(x/10**3))
                    else: formatted_labels.append('{:.0f}'.format(x/10**3))
            else:
                for i , x in enumerate(labels):
                    if i == 0 : formatted_labels.append('{:.2f}'.format(x))
                    elif i == len(labels)-1: formatted_labels.append('{:.2f}'.format(x))
                    else: formatted_labels.append('{:.0f}'.format(x))
                    
            print('value_divided spread',formatted_labels)

            cbar.set_ticklabels(formatted_labels)
        
        if save: plt.savefig("doc/img/" + save, bbox_inches='tight',pad_inches = 0)
        
        plt.show()
        plt.close()
        
def plot_histogram(x,color):
    x = x.flatten()
    avg = np.mean(x)
    print("Average",avg)

    plt.hist(x, density=True, bins=range(0,256),color = color, lw=0)
    plt.ylabel('Probability')
    plt.xlabel('Data')
    plt.yscale('log')
    #plt.ylim(0, 0.01)
    plt.show()
    plt.close()
    