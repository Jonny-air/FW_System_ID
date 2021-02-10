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
model_name = "EasyGlider3_210119_false" #<name>log_name_<committdateYYMMDD>_<statelevel=true,false>
log_name = 'sysid_ezg3_19012021'
log_type = 0    #0 = Lift/Drag, 1 = Throttle , 2 = Evaluation Data
log_location = './log'
# state_level = False   # is obtained from model parameters
filter_alpha = 0     # 0 no filter, 1 median filter (recommended), 2 moving average filter
filter_size = 7         # recommend 7 for median filter
ramp_type = 0        #0 manual ramp (or throttle), 1 elevator ramp, 2 pitch ramp
start_min = 4
start_sec = 52
end_min = 9
end_sec = 48

#Other Setup Parameters
override_logs = True
verbose = True
show_plots = True
rho = 1.222
g = 9.81

def add_log(verbose, show_plots):
    #SETUP
    csv_path = f"./csv/{model_name}_{log_type}.csv"
    start_time = start_min * 60 + start_sec
    end_time = end_min * 60 + end_sec

    #get parameters
    [m, n_0, n_slope, thrust_incl, D_p, actuator_control_ch, elev_control_ch, state_level] = get_params(model_name, verbose)
    # import log to dataframe
    if not os.path.isfile(f'./csv/{log_name}_vehicle_attitude_0.csv'):
        if verbose: print("Converting from ulog to csv...")
        convert(log_name)
    AttitudeDF = pd.read_csv(f'./csv/{log_name}_vehicle_attitude_0.csv', index_col=0, usecols=[0, 1, 2, 3, 4, 5, 6, 7])
    PosDF = pd.read_csv(f'./csv/{log_name}_vehicle_local_position_0.csv', index_col=0, usecols=[0, 10, 11, 12])
    AccelDF = pd.read_csv(f'./csv/{log_name}_sensor_accel_0.csv', index_col=0, usecols=[0, 3, 4, 5])
    WindDF = pd.read_csv(f'./csv/{log_name}_wind_estimate_0.csv', usecols=[0, 1, 2, 3])

    MergedDf = pd.merge_asof(PosDF, AttitudeDF, left_index=True, right_index=True)
    MergedDf = pd.merge_asof(MergedDf, AccelDF, left_index=True, right_index=True)

    ActuatorDF = pd.read_csv(f'./csv/{log_name}_actuator_controls_0_0.csv', index_col=0, usecols=[0, 2, 3, 4, 5, 6, 7, 8, 9])
    MergedDf = pd.merge_asof(MergedDf, ActuatorDF, left_index=True, right_index=True)

    # only take lines we want
    MergedDf = MergedDf.query(f'timestamp >= {start_time * 1.0E6} and timestamp <= {end_time * 1.0E6}')

    velocities = np.empty((0, 3))
    vas = np.empty((0, 1))
    fpas = np.empty((0, 1))
    atts = np.empty((0, 4))
    rolls = np.empty((0, 1))
    alphas = np.empty((0, 1))
    va_dots = np.empty((0, 1))
    fpa_dots = np.empty((0, 1))
    pitches = np.empty((0,1))
    elevs = np.empty((0, 1))
    if log_type == 1 or log_type == 2:
        n_ps = np.empty((0, 1))


    it = 0  # is there a better way to do this?
    for index, row in MergedDf.iterrows():
        # attitude quaternion
        att = row['q[0]'], row['q[1]'], row['q[2]'], row['q[3]']
        # Rotation Matrix from body frame to local frame
        R, roll, pitch, yaw = find_rotation(att)
        if log_type != 2:
            if roll>3*np.pi/180: continue
            if log_type == 0 and row['pitchspeed'] > 5*np.pi/180 : continue
            if log_type == 0 and row[f'control[{actuator_control_ch}]'] > 0.01 : continue
            if log_type == 2 and row[f'control[{actuator_control_ch}]'] < 0.1: continue
        #check take the clostest wind estimate to the current timestamp
        clostest_wind_index = WindDF['timestamp'].sub(index).abs().idxmin()
        wN =WindDF.loc[clostest_wind_index, 'windspeed_north']
        wE =WindDF.loc[clostest_wind_index, 'windspeed_east']
        wD =WindDF.loc[clostest_wind_index, 'windspeed_down']

        #air-mass-relative velocity in inertial frame
        velocities = np.append(velocities, [[row['vx']-wN, row['vy']-wE, row['vz']-wD]], axis=0)
        #velocity in body frame
        v_x, v_y, v_z = np.linalg.inv(R)@velocities[it,:] #forward, right, down
        #neglect sidewards component for airspeed
        va = np.sqrt(v_x**2 + v_z**2)
        if va == 0:
            continue #avoid division by zero down the line
        vas = np.append(vas, [[va]], axis=0)  # don't consider sideslip for airspeed

        pitches = np.append(pitches, [[pitch]], axis=0)
        elevs = np.append(elevs, [[float(row[f'control[{elev_control_ch}]'])]], axis=0)
        rolls = np.append(rolls, [[roll]], axis=0)
        # forward_vector = np.array([1.0 * np.cos(yaw), 1.0 * np.sin(yaw), 0.0])
        # right_vector = np.array([-1.0 * np.sin(yaw), 1.0 * np.cos(yaw), 0.0])
        # forward component of airspeed
        # va_f = project(velocities[it,:], forward_vector)
        # flight path angles
        # yaw = np.arctan2(R[1,0], 1.0-2*(row['q[0]']**2 + row['q[1]']**2)) #atan2(2.0 * (q.q3 * q.q0 + q.q1 * q.q2) , - 1.0 + 2.0 * (q.q0 * q.q0 + q.q1 * q.q1))

        fpas = np.append(fpas, [[np.arcsin(-row['vz']/ va)]], axis=0)  # fpa = arcsin(vz/va) in rad

        # alpha
        alphas = np.append(alphas, [pitch - fpas[it]], axis=0)
        # va_dots
        body_accel = np.array([row['x'], row['y'], row['z']])
        g_accel = np.linalg.inv(R) @ [0,0,g]
        if state_level:
            body_accel = body_accel / 1000
        else:
            g_accel = np.linalg.inv(R) @ [0, 0, g]
            body_accel += g_accel

        va_dots = np.append(va_dots, [np.cos(alphas[it])*body_accel[0] + np.sin(alphas[it])*body_accel[2]], axis=0) #airspeed change as change of total air mass reltive speed to avoid having attitude in here
        # fpa_dots
        # fpa_dot_vec = - np.cross(velocities[it, :], np.transpose(right_vector))
        fpa_dots = np.append(fpa_dots, [(-np.sin(alphas[it])*body_accel[0] + np.cos(alphas[it])*body_accel[2])], axis=0) #this is actually fpa_dot*va
        if log_type == 1 or log_type == 2:
            # thrust
            u_t = float(row[f'control[{actuator_control_ch}]'])
            n_p = n_0 + u_t * n_slope
            n_ps = np.append(n_ps, [[n_p]], axis=0)
        # increase counter
        it += 1
    if verbose:
        print("Data imported")

    if filter_alpha == 1:
        alphas_old = np.copy(alphas)
        # median filter
        median_filter(alphas, size=(filter_size, 1), output=alphas)
        if verbose:
            print("Alpha median filtered")

    if filter_alpha == 2:
        alphas_old = np.copy(alphas)
        # moving average filter
        n = filter_size
        crop = np.floor(n/2.0)
        ret = np.cumsum(alphas, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        alphas = ret[n - 1:] / n
        while crop > 0.0:
            alphas = np.append([alphas[0]], alphas, axis=0) #stretch in beginning
            alphas = np.append(alphas, [alphas[-1]], axis=0) #stretch in end
            crop -= 1
        if verbose:
            print("Alpha moving average filtered")

    if filter_alpha > 0 and show_plots:
        t = range(0, fpas.shape[0], 1)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot()
        ax.plot(t, alphas_old * 180 / np.pi, 'b', label="Unfiltered alphas [deg]")
        if filter_alpha == 1:
            ax.plot(t, alphas * 180 / np.pi, 'g', label="Median filtered alphas [deg]")
        if filter_alpha == 2:
            ax.plot(t, alphas * 180 / np.pi, 'g', label="Moving Average filtered alphas [deg]")
        ax.set_xlabel('datapoints')
        ax.set_ylabel('angle in deg')
        ax.legend(loc="best")
        plt.show()

    # check if model already exists

    if log_type == 1 or log_type == 2:
        newdata = np.column_stack((vas, va_dots, fpas, fpa_dots, alphas, rolls, pitches, elevs, n_ps))
        newDF = pd.DataFrame(data=newdata, columns=["vas", "va_dots", "fpas", "fpa_dots", "alphas", "rolls", "pitches", "elevs", "n_ps"])
    else:
        newdata = np.column_stack((vas, va_dots, fpas, fpa_dots, alphas, rolls, pitches, elevs))
        newDF = pd.DataFrame(data=newdata, columns=["vas", "va_dots", "fpas", "fpa_dots", "alphas", "rolls", "pitches", "elevs"])
    # add some additional information for later reference
    newDF["ramp_types"] = ramp_type
    newDF["log_names"] = log_name

    if os.path.isfile(csv_path):
        print("Trying to append to existing csv")
        # import existing csv to dataframe
        oldDF = pd.read_csv(csv_path, index_col=0)
        if oldDF.isin([f'{log_name}']).any().any():
            if not override_logs:
                print("[ERROR] A log with this name was already added")
                return  0
            else:
                print("Overriding existing log")
                clearIndexes = oldDF[oldDF['log_names'] == log_name].index
                oldDF.drop(clearIndexes, inplace=True)
        newDF = newDF.append(oldDF, ignore_index=True)

    #write csv
    newDF.to_csv(csv_path)
    print(f"All done, the csv file is in {csv_path}")
    return 0

if __name__ == '__main__':
    add_log(verbose, show_plots)