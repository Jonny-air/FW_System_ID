import os
import numpy as np
import pandas as pd
from Quat_to_Euler import find_rotation
from Project_vector import project
from scipy.ndimage import median_filter
from load_model import get_params
import matplotlib.pyplot as plt
from Convert_Log import convert

#Log Parameters
log_name = 'log_impact_1'
plt.rcParams.update({'font.size': 13})
log_location = './other_logs'
_xlim = None
_xlim = [56.5, 61.5]
_ylim = [-5, 5]

if (not os.path.isfile(f"./csv/{log_name}_sensor_accel_0.csv")):
    print("CSV does not exists yet - converting from log")
    convert(log_name, log_folder=log_location)
    # convert into dataframes
MergedDF = pd.read_csv(f'./csv/{log_name}_sensor_accel_0.csv')

acc_x = np.array([MergedDF.loc[:, 'x'].values]).transpose()/1000
acc_y = np.array([MergedDF.loc[:, 'y'].values]).transpose()/1000
acc_z = -np.array([MergedDF.loc[:, 'z'].values]).transpose()/1000
# t = range(acc_x.shape[0])
t = MergedDF['timestamp'].to_numpy() * 1.0E-6


def shift_time(time, start_time):
    time -= start_time

shift_time(t, t[0])

if _xlim is not None:
    shift_time(t, _xlim[0])
    _xlim[1] -= _xlim[0]
    _xlim[0] = 0


fig, axs = plt.subplots(1, figsize=(7,5))

axs.plot(t, acc_x, 'b', label='X acceleration')
axs.plot(t, acc_y, 'g', label='Y acceleration')
axs.plot(t, acc_z, 'm', label="Z acceleration")

for ax in [axs]:
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r'Acceleration [$\frac{m}{s^2}$]')
    ax.grid()
    ax.legend(loc='best')
    ax.set_xlim(_xlim[0]-5, _xlim[1])
    ax.set_ylim(_ylim)

plt.tight_layout()
plt.show()