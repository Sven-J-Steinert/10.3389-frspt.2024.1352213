from multiprocessing import Pool
import xarray as xr
import numpy as np
import sys
from ast import literal_eval
import random


########################################################################################################
#         PHYSICAL SIMULATION FOR ROCKET LAUNCH SEGEMENT          - to be called by multiprocessing
########################################################################################################

def f(x):
    return x*x




########################################################################################################


def save_to_netcdf(data,Longitude,Latitude,copy=False):
    
    xA = xr.DataArray(
            data=data,
            dims=["lat","lon"],
            coords=dict(
                lon=(["lon"], Longitude),
                lat=(["lat"], Latitude)
            ),
            attrs=dict(
                description="DeltaV to Gateway",
                var_desc="deltaV",
                units="m/s",
            ),
        )
    if copy:
        xA.to_netcdf("maps/Launch_Segement_copy" + str(random.randint(100000, 999999)) + ".nc")
    else:
        xA.to_netcdf("maps/Launch_Segement.nc")

def map_data(data_flat,Longitude,Latitude):
    
    data = np.zeros((len(Latitude), len(Longitude)))
    
    for Lat_count, Lat in enumerate(Latitude):
        for Lon_count, Lon in enumerate(Longitude):
            
            data[Lat_count][Lon_count] = data_flat[Lat_count*len(Longitude)+Lon_count]
    
    return data


if __name__ == '__main__':
    
    # fetch arguments
    try:
        arg = literal_eval(sys.argv[1])
    except:
        print('ERROR! argument required to run. shape: [Lon,Lat,init]')
        sys.exit()
        
    #print('Argument:',arg)
    
    Longitude = np.array(arg[0], dtype=np.short)
    Latitude  = np.array(arg[1], dtype=np.short)
    init_value = arg[2]

    #######################
    # RUN MULTIPROCESSING #
    #######################
    with Pool(5) as p:
        result_flat = p.map(f, init_value)
        
    result = map_data(result_flat,Longitude,Latitude)
    
    try:
        save_to_netcdf(result,Longitude,Latitude)
        print("Done.")
    except:
        print('ERROR! saving failed, check if file is in use.')
        save_to_netcdf(result,Longitude,Latitude,copy=True)
        print('BACKUP saving as copy.')