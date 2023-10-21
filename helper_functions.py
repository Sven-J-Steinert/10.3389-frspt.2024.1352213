# imports

import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm.notebook import tqdm

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

# globar variables

r_moon = 1737400    # [m] volumetric mean radius of the moon = reference height from heightmap
m_moon = 0.07346e24 # [kg] mass of the moon
G = 6.67430e-11 # Gravitational constant
g_0 = 9.80665 # [kg/s²] standard gravity

# helper functions

def info(dataset):
    print(f'{dataset.dtype} {dataset.shape} data range: {np.min(dataset)} - {np.max(dataset)}')

def LatLonfromShape(shape):
    Lat = np.zeros(shape[0])
    for i in range(shape[0]):
        Lat[i] = 90 - i * 180/len(Lat) - 0.5* 180/len(Lat)

    Lon = np.zeros(shape[1])
    for i in range(shape[1]):
        Lon[i] = 180 - i * 180/len(Lon) - 0.5* 180/len(Lon)
    Lon = np.flip(Lon)
    return Lat, Lon

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

        
def plot_map(values,value_devider,value_label,Lat_range,Lon_range,labelsize=None,save=None,bw=False,dpi=200,mass=False,labels=None,cmap='viridis',interpolation=None,return_data=False,center_zero=False,i_steps=None,silent=False):
    
        if not labelsize: labelsize = 20
    
        Lat_min, Lat_max = Lat_range
        Lon_min, Lon_max = Lon_range
        
        min_value = np.min(values)
        max_value = np.max(values)
        
        if not silent: print("display values",values.shape[::-1])
        if not silent: print("extrema",min_value,max_value)
        
        plt.figure(figsize=(12,6), dpi=dpi)
        
        ax = plt.gca()

        if bw: im = ax.imshow(values, cmap='gray', interpolation=str(interpolation), extent=[Lon_min,Lon_max,Lat_min,Lat_max])
        else:
            im = ax.imshow(values, cmap=cmap, interpolation=str(interpolation), extent=[Lon_min,Lon_max,Lat_min,Lat_max])
            
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

            # to have suitable ticks on cbar: find range from  [min .. max]
            # devide it into steps 1 order lower than difference results in ~10 ticks
            
            delta = (max_value/value_devider) - (min_value/value_devider)
            if not silent: print('orderOfMagnitude(delta)',orderOfMagnitude(delta))
            maximum_flat_tick = 10**orderOfMagnitude(delta)
            one_step = (maximum_flat_tick/10) * value_devider
            min_flat_value = 10**orderOfMagnitude(min_value/value_devider)*value_devider
            if not silent: print(f'min_flat_value {min_flat_value}')
            
            if not mass:
                if not silent: print('inside normal bottom scaling')
                inter_steps = np.arange(min_value, max_value, one_step)[1:]
                if not silent: print('inter_steps',len(inter_steps),inter_steps)
                if len(inter_steps) > 15:
                    inter_steps = np.delete(inter_steps, np.arange(0, inter_steps.size, 2)) # delete every second step
                if not silent: print('inter_steps',len(inter_steps),inter_steps)
                if center_zero: inter_steps = np.append(inter_steps, 0) # add Zero
                if i_steps is not None: inter_steps = i_steps
                labels = np.append( np.append(min_value, inter_steps), max_value)
            else:
                if not silent: print('inside weird bottom scaling')
                while min_flat_value - (min_value + one_step) < 0 : min_flat_value += one_step
                if not silent: print('max_value',max_value)
                labels = np.append(np.append(min_value, np.arange(min_flat_value, max_value, one_step )), max_value)
                if not silent: print(labels)
                

            if not silent: print('image value spread',labels)
            loc = labels
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
                    
            if not silent: print('value_divided spread',formatted_labels)

            cbar.set_ticklabels(formatted_labels)
            
            ax.set_xlabel('Longitude $\lambda \ [\mathrm{deg}]$',fontsize=labelsize)
            ax.set_ylabel('Latitude $\phi \ [\mathrm{deg}]$',fontsize=labelsize)
        
        if save: plt.savefig("doc/img/" + save, bbox_inches='tight',pad_inches = 0)
        
        plt.show()

        if return_data: # return interpolated values if asked
            #plt.imshow(values, interpolation=str(interpolation), cmap='viridis')
            #interpolated_data = plt.gci().get_array()
            return ax.get_images()[0].get_array() 
            
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
    
    
def plot_launch_segment(df):
    phase1_end = 10
    phase2_end = 15
    phase3_end = 20
    phase4_end = len(df)

    # plot path cartesian
    dpi = 72

    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10),dpi=dpi)

    # Create 2x2 sub plots
    gs = gridspec.GridSpec(ncols=3, nrows=1,width_ratios=[1,20,2.5])

    pl.figure(figsize=(40,10))


    # left
    ax = pl.subplot(gs[0, 0]) # row 0, col 0

    offset_y = r_moon
    # moon surface
    t = np.linspace(0.49*np.pi,0.51*np.pi,1000)
    pl.plot(r_moon*np.cos(t), r_moon*np.sin(t)-offset_y,color='grey', linewidth=2)

    xpoints = df.loc[0.0:phase3_end+2]['pos_x [m]']
    ypoints = df.loc[0.0:phase3_end+2]['pos_y [m]']

    max_x = np.max(xpoints)
    max_y = np.max(ypoints-offset_y)
    min_x = np.min(xpoints)
    min_y = np.min(ypoints-offset_y)

    pad_y = 1e1
    pad_x = 1e2

    ax.set_xlim(min_x-pad_x, max_x)
    ax.set_ylim(min_y-pad_y, max_y+pad_y)

    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    pl.plot(xpoints[:round(phase1_end+1)],ypoints[:round(phase1_end+1)]-offset_y                                  , color='deeppink') # phase 1
    pl.plot(xpoints[round(phase1_end):round(phase2_end+1)],ypoints[round(phase1_end):round(phase2_end+1)]-offset_y, color='magenta') # phase 2
    pl.plot(xpoints[round(phase2_end):round(phase3_end+1)],ypoints[round(phase2_end):round(phase3_end+1)]-offset_y, color='darkviolet') # phase 3
    pl.plot(xpoints[round(phase3_end):],ypoints[round(phase3_end):]-offset_y                        ,color='tab:blue') # phase 4

    #middle
    ax = pl.subplot(gs[0, 1])

    t = np.linspace(0.45*np.pi,0.51*np.pi,1000)
    pl.plot(r_moon*np.cos(t), r_moon*np.sin(t)-offset_y,color='grey', linewidth=2)

    xpoints = df.loc[phase3_end:phase4_end]['pos_x [m]']
    ypoints = df.loc[phase3_end:phase4_end]['pos_y [m]']

    max_x = np.max(xpoints)
    max_y = np.max(ypoints-offset_y)
    min_x = np.min(xpoints)
    min_y = np.min(ypoints-offset_y)

    pad_y = 1e4
    pad_x = 1e3

    ax.set_xlim(min_x-pad_x, max_x)
    ax.set_ylim(min_y-pad_y, max_y+pad_y)

    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    pl.plot(xpoints,ypoints-offset_y,color='tab:blue', linewidth=2) # phase 4

    xpoints = df.loc[:phase3_end]['pos_x [m]']
    ypoints = df.loc[:phase3_end]['pos_y [m]']

    pl.plot(xpoints,ypoints-offset_y, linewidth=2, color='magenta') # until phase 4

    # right
    ax = pl.subplot(gs[0, 2]) # row 0, col 1
    pl.plot([0,1])

    # moon surface
    t = np.linspace(0,2*np.pi,100)
    pl.plot(r_moon*np.cos(t), r_moon*np.sin(t),color='grey', linewidth=2)

    xpoints_prop = df.loc[phase4_end:]['pos_x [m]']
    ypoints_prop = df.loc[phase4_end:]['pos_y [m]']
    pl.plot(xpoints_prop,ypoints_prop, color='tab:green', linewidth=1) # propagate

    xpoints = df.loc[0.0:phase4_end]['pos_x [m]']
    ypoints = df.loc[0.0:phase4_end]['pos_y [m]']
    pl.plot(xpoints,ypoints, linewidth=2 ) # start

    ext = np.max(np.abs([ df['pos_x [m]'],df['pos_y [m]']]))

    ax.set_ylim(-ext, ext)
    ax.set_xlim(-ext, ext)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)

    plt.savefig('doc/img/launch_segment.png',bbox_inches='tight',dpi=300)
    #plt.axis('off')
    plt.show()

    # plot result properties graphs

    df[['altitude [m]','vel_r [m/s]','vel_phi [m/s]','acc_r [m/s²]','acc_phi [m/s²]','dir_n [°]']].plot(subplots=True,figsize=(25,20),grid=True,xlim=[0, df.index[-1]])
    #df.plot(subplots=True,figsize=(20,25))
    plt.show()
    

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))