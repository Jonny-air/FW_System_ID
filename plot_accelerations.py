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
log_name = 'log_altitude_2'
log_location = './other_logs'

if (not os.path.isfile(f"./csv/{log_name}_sensor_accel_0.csv")):
    print("CSV does not exists yet - converting from log")
    convert(log_name, log_folder=log_location)
    # convert into dataframes
MergedDF = pd.read_csv(f'./csv/{log_name}_sensor_accel_0.csv', index_col=0)

acc_x = np.array([MergedDF.loc[:, 'x'].values]).transpose()
acc_y = np.array([MergedDF.loc[:, 'y'].values]).transpose()
acc_z = np.array([MergedDF.loc[:, 'z'].values]).transpose()
t = range(acc_x.shape[0])
fig, axs = plt.subplots(3, figsize=(10, 10))

axs[0].plot(t, acc_x, 'b', label='X accelerations')
axs[1].plot(t, acc_y, 'b', label='Y accelerations')
axs[2].plot(t, acc_z, 'b', label="Z accelerations")

axs[0].set_xlabel('Datapoints')
axs[1].set_xlabel('Datapoints')
axs[2].set_xlabel('Datapoints')
axs[0].set_ylabel('m/s2')
axs[1].set_ylabel('m/s2')
axs[2].set_ylabel('m/s2')

axs[0].grid()
axs[1].grid()
axs[2].grid()

axs[0].legend(loc='best')
axs[1].legend(loc='best')
axs[2].legend(loc='best')
plt.show()