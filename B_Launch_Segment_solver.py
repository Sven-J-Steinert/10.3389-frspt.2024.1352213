from multiprocessing import Pool
import xarray as xr
import numpy as np
import sys
from ast import literal_eval
import random
from itertools import repeat


########################################################################################################
#         PHYSICAL SIMULATION FOR ROCKET LAUNCH SEGEMENT          - to be called by multiprocessing
########################################################################################################

# imports
import numpy as np
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
from io import StringIO
from csv import writer 
from scipy import integrate
import math
import numba
from numba import jit
import time as time_lib
from tqdm import tqdm
import xarray as xr
from multiprocessing import Pool

######


# global constants

r_moon = 1737400    # [m] volumetric mean radius of the moon = reference height from heightmap
m_moon = 0.07346e24 # [kg] mass of the moon
G = 6.67430e-11 # Gravitational constant



# acceleration direction from propulsion
#         0°
#         │                
# 270° ───┼─── 90°     reference angle from vertical axis, perpendicular to lunar surface
#         │
#        180°



def angle(vec,v_ref=None,rad=False):
    
    if v_ref is not None:
        vector_1 = v_ref
    else:
        vector_1 = [0, 1, 0]
    vector_2 = vec
    
    if np.linalg.norm(vector_2) == 0: return None

    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    
    if vec[0]<0:
        angle = 2*np.pi - angle
    
    if rad: return angle
    return np.rad2deg(angle)
    

def vec_from_angle(deg): # in [x,y] cartesian-coordinates
    rad = np.deg2rad(90-deg)

    y = np.sin(rad)
    x = np.cos(rad)
    vec = np.array([x,y,0], dtype=np.float64)
    
    return vec

def dir_from_angle(deg): # in [x,y] cartesian-coordinates
    if deg >= 0:
        rad = np.deg2rad(90-deg)
    if deg < 0:
        rad = - np.deg2rad(90-abs(deg))

    y = np.sin(rad)
    x = np.cos(rad)
    vec = np.array([x,y,0], dtype=np.float64)
    
    return vec

def plot_vector(vec):
    
    plt.figure(figsize=(3, 3), dpi=50)
    
    # reference vector
    xpoints = np.array([0, 0])
    ypoints = np.array([0, 1])
    plt.plot(xpoints, ypoints, '--', color = 'b')
    
     
    xpoints = np.array([0, vec[0]])
    ypoints = np.array([0, vec[1]])
        
    plt.plot(xpoints, ypoints, color = 'r')
    
    plt.ylim(-1.1, 1.1)
    plt.xlim(-1.1, 1.1)
    
    plt.gca().set_aspect('equal', adjustable='box')
    
    #unit circle
    t = np.linspace(0,np.pi*2,100)
    plt.plot(np.cos(t), np.sin(t), linewidth=2)
    
    plt.axis('off')
    plt.show()


def r_sys(vec, pos):

    alpha = angle(pos,rad=True)
    #print(alpha)

    rot = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    #print(rot)

    v = np.array([vec[0], vec[1]], dtype=np.float64)

    v2 = np.dot(rot, v)
    
    r_vec = np.array([v2[0], v2[1], vec[2]], dtype=np.float64)
    #print(vec,r_vec)
    
    return r_vec

def xy_sys(vec, pos):

    alpha = -angle(pos,rad=True) 
    #print(alpha)

    rot_xy = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    #print(rot)

    v = np.array([vec[0], vec[1]], dtype=np.float64)

    v2 = np.dot(rot_xy, v)
    
    xy_vec = np.array([v2[0], v2[1], vec[2]], dtype=np.float64)
    #print(vec,r_vec)
    
    return xy_vec

def altitude(pos):
    r = np.linalg.norm(pos)
    return r - r_moon

def deltaV(mass_fuel_used,mass_full,EEV):
    # compute deltaV from used mass of fuel
    dV = EEV * np.log(mass_full / (mass_full-mass_fuel_used))
    
    return dV

def downrange_distance(pos):
    # known starting point at [ x = 0, y = r_moon ]
    alpha = angle(pos)
    # circular arc = alpha * r
    dist = alpha * r_moon
    return dist

@jit(nopython=True)
def orbital_velocity(altitude):
    return np.sqrt((G * m_moon)/(r_moon + altitude))

@jit(nopython=True)
def acc_grav_moon(pos):
    r = np.linalg.norm(pos)
    r_val = G * (m_moon/(r**2))
    e_r = -(pos / r)
    
    return (e_r * r_val)

def acc_centrifugal(pos,vel,scalar=False):
    r = np.linalg.norm(pos)
    v = r_sys(vel,pos)[0] # vel portion that is ⊥ to r -> phi_val
    r_val = (v**2)/(r)
    if scalar: return r_val
    else:
        e_r = (pos / r)
        return (e_r * r_val)

@jit(nopython=True)
def acc_propulsion(m,m_flow,EEV):
    F_thrust = m_flow * EEV
    acc = F_thrust / m

    return acc
    

class PID_Controller:
  def __init__(self, set_alt , t_step, p_val, i_val, d_val):
    
    # variables
    self.int_sum = 0
    self.e_old = None
    self.t_step = t_step
    self.set_alt = set_alt
    self.p_val = p_val # 2
    self.i_val = i_val # 0.1
    self.d_val = d_val # 10
    
  def step(self,pos):
    # calculate error

    e = self.set_alt - altitude(pos)
    
    # simple integration
    self.int_sum += e * self.t_step    
    

    if self.e_old is None:
        d_val = 0
    else:
        d_val = (e - self.e_old) / self.t_step
    
    self.e_old = e
    
    # Controller formular
    u =  self.p_val *e + self.d_val * d_val + self.i_val * self.int_sum
    
    # sigmoid for mapping
    try:
        exp = math.exp(-0.001 *u)
    except OverflowError:
        exp = float('inf')
    
    sigm = (2/(1 + exp) ) -1
    
    # +/-60° freedom
    freedom = 20 # [°] in r_sys
    alpha = freedom * sigm

    dir_n = dir_from_angle(90-alpha)
    
    # transform from r_sys to x_sys
    dir_n_xy = xy_sys(dir_n, pos)

    return dir_n_xy


@jit(nopython=True)
def phys_sim_step(time,t_step, pos, vel, acc, m, mass_dry, m_flow, EEV, dir_n, engine_on):
    
    # one sim step BEGIN
    # Velocity Verlet - numerical solution to differential equation
    ###############################
    new_pos = np.add( np.add(pos, vel*t_step) , acc*(t_step*t_step*0.5))
    new_acc = np.add( dir_n * acc_propulsion(m,m_flow,EEV), acc_grav_moon(pos))
    new_vel = np.add( vel , (np.add(acc,new_acc))*(t_step*0.5))
    
    if engine_on:
        new_mass = np.subtract(m , (m_flow * t_step))
        if new_mass < mass_dry:
            print('CRITICAL - FUEL EMPTY')
            new_mass = m
            engine_on = False
    else: new_mass = m
    
    new_time = np.around(time + t_step,3) # digit_precision = 3
    
    return new_time,new_pos,new_vel,new_acc,new_mass,engine_on ;

    ###############################
    # one sim step END


def write_data_row(csv_writer,time,pos,vel,acc,m,dir_n):
    csv_writer.writerow([time,m,altitude(pos),pos[0],pos[1],pos[2],r_sys(vel,pos)[1],r_sys(vel,pos)[0],vel[0],vel[1],vel[2],r_sys(acc,pos)[1],r_sys(acc,pos)[0],acc[0],acc[1],acc[2],angle(dir_n)])
    
def log(csv_writer,time,pos,vel,acc,m,dir_n):
    x = 1
    # logging results only every x seconds
    if (time*1000)%(x) == 0:
        write_data_row(csv_writer,time,pos,vel,acc,m,dir_n)
        
def f_lin_p_val(start_altitude):
    val_min = 1.96     # for lowest altitude -9115
    val_max = 16.878    # for highest altitude 10775
    dx = 10775 + 9115
    dy = val_max - val_min
    n = val_max - (dy/dx) * 10775 # plugged point in
    
    lin_interpol = (dy/dx) * start_altitude + n
    
    return lin_interpol

def f_lin_d_val(start_altitude):
    val_min = 60     # for lowest altitude -9115
    val_max = 300    # for highest altitude 10775
    dx = 10775 + 9115
    dy = val_max - val_min
    n = val_max - (dy/dx) * 10775 # plugged point in
    
    lin_interpol = (dy/dx) * start_altitude + n
    
    return lin_interpol

# SIMULATION
#@jit(nopython=False,forceobj=True)
def simulation(t_step,start_altitude,target_altitude,mass_dry,mass_full,m_flow,single_mode):

    # set launch carrier properties
    # EL3 - European Large Logistics Lander
    mass_fuel = mass_full - mass_dry # kg
    I_sp = 400 # [s] specific impulse
    g_0 = 9.80665 # [kg/s²] standard gravity
    EEV = np.array(I_sp * g_0, dtype=np.float64) # [m/s] effective exhaust velocity

    # setup parameters
    t_step = np.array(t_step, dtype=np.float64) # [s] time for one simulation step
    #digit_precision = 3

    time = np.array(0, dtype=np.float64)

    # cartesian coordinates - 3D
    #         y   z
    #         │ /              
    #      ───┼─── x    [x,y,z]
    #       / Moon       
    # point of origin = moon center mass point


    # initialize
    # Spacecraft properties

    # mass
    m = mass_full  # [kg]

    # position
    pos = np.array([0,r_moon + start_altitude,0], dtype=np.float64) # [m]

    # velocity
    #add ground speed vel
    vel = np.array([0,0,0], dtype=np.float64) # [m/s]

    # acceleration
    acc = np.array([0,0,0], dtype=np.float64) # [m/s²]

    # engine toggle
    engine_on = True
    # engine acc direction
    dir_n = vec_from_angle(0)

    # pandas df for logging
    # to increase appending speed, rows are written into memory then read back in pandas df
    output = StringIO()
    csv_writer = writer(output)

    csv_writer.writerow(['time [s]','mass [kg]','altitude [m]','pos_x [m]','pos_y [m]','pos_z [m]','vel_r [m/s]','vel_phi [m/s]','vel_x [m/s]','vel_y [m/s]','vel_z [m/s]','acc_r [m/s²]','acc_phi [m/s²]','acc_x [m/s²]','acc_y [m/s²]','acc_z [m/s²]','dir_n [°]'])

    csv_writer.writerow([time,m,altitude(pos),pos[0],pos[1],pos[2],r_sys(vel,pos)[1],r_sys(vel,pos)[0],vel[0],vel[1],vel[2],r_sys(acc,pos)[1],r_sys(acc,pos)[0],acc[0],acc[1],acc[2],angle(dir_n)])
    #write_data_row(time,pos,vel,acc,m,dir_n) # inital values at index 0 at time 0


    # Simulation execution

    """
    print('''
    ┌───────────┬─────────────────────────────────┐
    │  Phase 1  │          Vertical rise          │
    └───────────┴─────────────────────────────────┘''')
    """
    dir_n = vec_from_angle(0)
    #plot_vector(dir_n)

    # stop condition: time
    # stop Apollo velocity: 50 [ft/s] = 15.24 [m/s]
    # stop Appolo height: ~250 [ft] = 76.2 [m]

    #while vel[1] < 15.24: 
    while time < 10: 

        
        time, pos, vel, acc, m, engine_on = phys_sim_step(time,t_step,pos,vel,acc,m,mass_dry, m_flow, EEV, dir_n,engine_on) # pysical sim step

        log(csv_writer,time,pos,vel,acc,m,dir_n) # log every x [ms]


    phase1_end = time
    """
    print(f'Phase 1 finished at {phase1_end} s, Apollo 10 [s]')
    print(f'> height {pos[1]-r_moon} m, Apollo 76.2 [m]')
    print(f'> vertical vel {vel[1]} m/s Apollo 15.24 [m/s]')

    print('''
    ┌───────────┬─────────────────────────────────┐
    │  Phase 2  │      first tilt, 38° burn       │
    └───────────┴─────────────────────────────────┘''')
    """
    dir_n = vec_from_angle(38)
    #plot_vector(dir_n)

    # stop condition: time
    time_now = time
    while time < time_now + 2: 

        time, pos, vel, acc, m, engine_on = phys_sim_step(time,t_step,pos,vel,acc,m,mass_dry, m_flow, EEV, dir_n,engine_on) # pysical sim step

        log(csv_writer,time,pos,vel,acc,m,dir_n) # log every x [ms]

    phase2_end = time 
    """
    print(f'Phase 2 finished at {phase2_end} s')

    print('''
    ┌───────────┬─────────────────────────────────┐
    │  Phase 3  │     second tilt, 52° burn       │
    └───────────┴─────────────────────────────────┘''')
    """
    dir_n = vec_from_angle(52)
    #plot_vector(dir_n)

    # stop condition: time
    time_now = time
    while time < time_now + 2: 

        time, pos, vel, acc, m, engine_on = phys_sim_step(time,t_step,pos,vel,acc,m,mass_dry, m_flow, EEV, dir_n,engine_on) # pysical sim step

        log(csv_writer,time,pos,vel,acc,m,dir_n) # log every x [ms]

    phase3_end = time
    
    
    
    alt = altitude(pos)
    # init Controller to hold current altitude
    PID = PID_Controller(target_altitude, t_step, f_lin_p_val(start_altitude), 0.000000001, f_lin_d_val(start_altitude)) #altitude(pos)
    
    """
    print(f'> current Altitude {alt} m')
    print('''
    ┌────────────┐
    │ Controller │ holding altitude constant with PID-Regulator
    └────────────┘''')
    """

    # stopping condition: orbital velocity (vel_phi)
    while r_sys(vel,pos)[0] < orbital_velocity(alt):

        dir_n = PID.step(pos)
        
        time, pos, vel, acc, m, engine_on = phys_sim_step(time,t_step,pos,vel,acc,m,mass_dry, m_flow, EEV, dir_n,engine_on) # pysical sim step
        
        if not engine_on: break

        alt = altitude(pos)

        log(csv_writer,time,pos,vel,acc,m,dir_n) # log every x [ms]

    phase4_end = time
    """
    print(f'Phase 4 finished at {phase4_end} s')
    print(f'> fuel mass used until here {mass_full - m} kg')
    print(f'> Orbital velocity {r_sys(vel,pos)[0]} m/s reached, where {orbital_velocity(alt)} m/s is needed for altitude {alt}')
    print(f'> pos {pos} m reached')
    print(f'> current Orbital velocity {r_sys(vel,pos)[0]} m/s')
    print(f'> Altitude {alt} m reached')
    """

    fuel_used = mass_full - m
    deltaV_result = deltaV(fuel_used,mass_full,EEV)
    downrange_result = (downrange_distance(pos)/1000)

    
    result_body = f'{{ "time":{time}, "fuel_used":{fuel_used}, "deltaV":{deltaV_result}, "downrange_distance": {downrange_result}, "altitude":{altitude(pos)}, "vel_phi":{r_sys(vel,pos)[0]},"vel_r":{r_sys(vel,pos)[1]} }}'


    """
    print('''
    ┌───────────┬─────────────────────────────────┐
    │  Phase 5  │   orbit propagate, engine off   │
    └───────────┴─────────────────────────────────┘''')

    dir_n =  np.array([0,0,0], dtype=np.float64)
    #plot_vector(dir_n)
    engine_on = False

    now_time = time

    while time < now_time + 400:

        phys_sim_step()

        log(1000) # log every full x [ms]

    
    
    print('''
    ├───────────────── FINISHED ──────────────────┤
    ''')
    """


    # load pandas df back from memory
    output.seek(0) # we need to get back to the start of the StringIO
    df = pd.read_csv(output)
    df = df.set_index('time [s]')
    
    
    # plotting path
    if False:
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
    
    return result_body, df


def bind_simulation(start_altitude,target_altitude, t_step, mass_full, m_flow, single_mode=False):

    result_body, df = simulation( t_step=t_step,
                                   start_altitude=start_altitude,
                                   target_altitude=target_altitude,
                                   mass_dry=1600,
                                   mass_full=mass_full,
                                   m_flow=m_flow,
                                   single_mode=single_mode)
        
    
    return result_body

########################################################################################################


def save_to_netcdf(data,Longitude,Latitude,filename,copy=False):
    
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
        xA.to_netcdf( filename + "_copy" + str(random.randint(100000, 999999)) + ".nc")
    else:
        xA.to_netcdf( filename + ".nc")

def map_data(data_flat,Longitude,Latitude):
    
    data = np.zeros((len(Latitude), len(Longitude)))
    
    for Lat_count, Lat in enumerate(Latitude):
        for Lon_count, Lon in enumerate(Longitude):
            
            data[Lat_count][Lon_count] = data_flat[Lat_count*len(Longitude)+Lon_count]
    
    return data


if __name__ == '__main__':
    
    # fetch arguments
    try:
        filename = sys.argv[1]
    except:
        print('ERROR! arguments required, call -h for help')
        sys.exit()
        
    if filename in ['-h','-H','-help','help']:
        print('printing help')
        sys.exit()
    
    try:
        arg = literal_eval(sys.argv[2])
    except:
        print('ERROR! argument required to run. shape: [Lon,Lat,init]')
        print(literal_eval(sys.argv[2]))
        sys.exit()
        
    
    Longitude = np.array(arg[0], dtype=np.short)
    Latitude  = np.array(arg[1], dtype=np.short)
    t_step = arg[2]
    mass_full = arg[3]
    start_altitude = arg[4]
    target_altitude = arg[5]
    m_flow = arg[6]
    
    """
    print(f'filename {filename}')
    print(f'Longitude {Longitude}')
    print(f'Latitude {Latitude}')
    print(f't_step {t_step}')
    print(f'mass_full {mass_full}')
    print(f'start_altitude {start_altitude}')
    """
    single_mode = False
    if filename == "single_mode":
        single_mode = True
        
    if single_mode:
        #######################
        # RUN  SINGLE-MODE    #
        #######################
        
        if isinstance(start_altitude, list): start_altitude = start_altitude[0]
        if isinstance(t_step, list): t_step = t_step[0]
        if isinstance(mass_full, list): mass_full = mass_full[0]
        
        print(bind_simulation(start_altitude,target_altitude,t_step,mass_full, m_flow, single_mode=True))
    
    else:
        #######################
        # RUN MULTIPROCESSING #
        #######################
        with Pool() as p:
            _ , result_flat = p.starmap(bind_simulation, zip(start_altitude,repeat(target_altitude), repeat(t_step), repeat(mass_full), repeat(m_flow)))

        result = map_data(result_flat,Longitude,Latitude)

        try:
            save_to_netcdf(result,Longitude,Latitude,filename)
            print("Done.")
        except:
            print('ERROR! saving failed, check if file is in use.')
            save_to_netcdf(result,Longitude,Latitude,filename,copy=True)
            print('BACKUP saving as copy.')