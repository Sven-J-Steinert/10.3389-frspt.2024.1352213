import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from PIL import Image, ImageTk, ImageMath
Image.MAX_IMAGE_PIXELS = 1000000000
import PIL.ImageGrab as ImageGrab
import cv2

from tqdm import tqdm   # progreess bar

import os

# IR  = Ilmenite Reduction
# MRE = Molten Regolith Electrolysis
# MSE = Molten Salt Electrolysis

# IR Settings
IR_efficiency = 0.75

class Simulation():
    def __init__(self, resolution, Lat_span, Lon_span):

        self.Lat_min, self.Lat_max = Lat_span
        self.Lon_min, self.Lon_max = Lon_span
        print("Simulation Area:", "Lon",tuple((self.Lon_min, self.Lon_max)),
                                  "Lat",tuple((self.Lat_min, self.Lat_max)))

        self.aspect_ratio = (self.Lon_max-self.Lon_min)/(self.Lat_max-self.Lat_min)
        self.resolution = tuple((resolution,int(resolution/self.aspect_ratio)))
        self.resolution_im = tuple((int((360/(self.Lon_max-self.Lon_min))*resolution),
                                    int((360/(self.Lon_max-self.Lon_min))*resolution/2)))
        print("Simulation Resolution", self.resolution)



        #self.plot_global_map(self.ilmenite_map,resolution,'TiO2 [wt%]','TiO2')

        # rescale all maps to calculation resolution
        TiO2_map = self.resize_map("WAC_TIO2_GLOBAL_MASKED_MAP.png")
        # map the shaped index for all maps

        # store image data in np array for calculation
        TiO2_values = self.read_im_values(im=TiO2_map,value_divider=22)
        # plot result
        self.plot_global_map(TiO2_values,'TiO2 [wt%]','TiO2')


    def resize_map(self,path):
        print("resize",path, 'to', self.resolution_im)
        image = Image.open("maps/" + path)
        print("original map image", "with extrema: ",image.getextrema())
        new_image = image.resize(self.resolution_im)
        new_image.save('temp.png')
        new_image = cv2.imread('temp.png',0)
        new_image[new_image<22] = 22 # set everything below 1% to 1 w%
        cv2.imwrite('temp.png', new_image)
        new_image = Image.open("temp.png")
        print("resized map image", "with extrema: ",new_image.getextrema())
        os.remove("temp.png")

        return new_image

    def read_im_values(self,im,value_divider):
        data = im.load()
        im_res = im.size

        Lat_use = np.linspace(self.Lat_max,self.Lat_min,self.resolution[1])
        print('Lat_use',np.min(Lat_use),np.max(Lat_use),Lat_use.shape, 'first Lat', Lat_use[0:5])
        Lon_use = np.linspace(self.Lon_min,self.Lon_max,self.resolution[0])
        print('Lon_use',np.min(Lon_use),np.max(Lon_use),Lon_use.shape, 'first Lon', Lon_use[0:5])

        print("     reading image values", "with extrema: ",im.getextrema())

        x = np.zeros(shape=(tuple((self.resolution[1],self.resolution[0]))))

        for n in tqdm(range(self.resolution[1])):

            index_y = round(np.interp(Lat_use[n], (-90,90), (im_res[1]-1,0)))

            for i in range(self.resolution[0]):
                index_x = round(np.interp(Lon_use[i], (-180, 180), (0, im_res[0]-1)))
                x[n][i] = data[index_x,index_y]/value_divider

        return x

    def solar_radiation_map(self,Lat,Lon):
        if peak_of_eternal_light_map(Lat,Lon):
            uptime_ratio = 0.8
        else:
            uptime_ratio = 0.5
        power = 1000
        return power, uptime_ratio

    def peak_of_eternal_light_map(self,Lat,Lon):
        if Lat > 85 or Lat < -85:
            return True
        else:
            return False

    def f_IR(self,Lat,Lon,amount_of_regolith):
        ilmenite_ratio = ilmenite_map(Lat,Lon)
        product = ilmenite_ratio*IR_efficiency*amount_of_regolith
        return product

    def f_MRE(self,Lat,Lon,amount_of_regolith):
        return x

    def f_MSE(self,Lat,Lon,amount_of_regolith):
        return x


    def plot_global_map(self,values,value_label,file_name):
        print("display values",values.shape[::-1])
        plt.figure(figsize=(12,6), dpi=100)
        ax = plt.gca()
        im = ax.imshow(values, cmap='viridis', interpolation='None',
                extent=[self.Lon_min,self.Lon_max,self.Lat_min,self.Lat_max])
        plt.xticks(np.arange(self.Lon_min, self.Lon_max+1, 20.0))
        plt.yticks(np.arange(self.Lat_min, self.Lat_max+1, 20.0))
        # create an axes on the right side of ax. The width of cax will be 2%
        # of ax and the padding between cax and ax will be fixed at 0.4 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.25, pad=0.4)

        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.set_ylabel(value_label, rotation=90)

        labels = np.arange(np.min(values),np.max(values),1)
        print('cbar labels',labels)
        loc    = labels
        cbar.set_ticks(loc)
        if labels[0] == round(labels[0]):
            cbar.set_ticklabels(['{:.0f}'.format(x) for x in labels])
        else:
            cbar.set_ticklabels(labels)

        plt.tight_layout()
        plt.savefig(file_name+'.svg')
        plt.show()
        plt.close()


instance = Simulation(resolution=1000,Lat_span=tuple((-90,90)),Lon_span=tuple((-180,180)))
