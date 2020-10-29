import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#setup
start_min = 1
start_sec = 2
end_min = 1
end_sec = 30
log_name = 'log_B2'

#parameters
g =9.81
m = 3.7
rho = 1.225
n_0 = 15.03 #radps
n_slope = 985
thrust_incl = 0.0 #rad
D_p = 0.28
CD = np.array(
    [0.02652086160037974, 0.16581006194228595, -0.8575806869624077]
)
CL = np.array(
    [0.14363326331809678, 1.6261793088854715]
)

#setup behind the curtains
start_time = start_min*60+start_sec
end_time = end_min*60+end_sec
AttitudeDF = pd.read_csv(f'./IDFILES/{log_name}_vehicle_attitude_0.csv', index_col=0, usecols=[0, 4, 5, 6, 7])
PosDF = pd.read_csv(f'./IDFILES/{log_name}_vehicle_local_position_0.csv', index_col=0, usecols=[0, 10, 11, 12])
AccelDF = pd.read_csv(f'./IDFILES/{log_name}_sensor_accel_0.csv', index_col=0, usecols=[0, 3, 4, 5])
ActuatorDF = pd.read_csv(f'./IDFILES/{log_name}_actuator_controls_0_0.csv', index_col=0, usecols=[0, 5])

MergedDf = PosDF.merge(AttitudeDF, left_index=True, right_index=True)
MergedDf = MergedDf.merge(AccelDF, left_index=True, right_index=True)
MergedDf = pd.merge_asof(MergedDf, ActuatorDF, left_index=True, right_index=True)

#only take lines we want
MergedDf = MergedDf.query(f'timestamp >= {start_time*1.0E6} and timestamp <= {end_time*1.0E6}')

def find_rotation(q):
    R = np.empty([3,3])

    aa = q[0]*q[0]
    ab = q[0]*q[1]
    ac = q[0]*q[2]
    ad = q[0]*q[3]
    bb = q[1]*q[1]
    bc = q[1]*q[2]
    bd = q[1]*q[3]
    cc = q[2]*q[2]
    cd = q[2]*q[3]
    dd = q[3]*q[3]

    roll = np.arctan2(2*(cd+ab), 1.0-2.0*(bb + cc)) #atan2(2.0 * (d * c + a * b) , 1.0 - 2.0 * (q.q1 * q.q1 + q.q2 * q.q2))
    pitch = np.arcsin(2*(ac-bd))
    yaw = np.arctan2(2*(bc+ad), -1.0+2.0*(aa + bb))

    R[0,0] = aa+bb-cc-dd
    R[0,1] = 2*(bc-ad)
    R[0,2] = 2*(bd+ac)
    R[1,0] = 2*(bc+ad)
    R[1,1] = aa-bb+cc-dd
    R[1,2] = 2*(cd-ab)
    R[2,0] = 2*(bd-ac)
    R[2,1] = 2*(cd+ab)
    R[2,2] = aa-bb-cc+dd

    return R, roll, pitch, yaw

def transform_to_local(vec, q): #vec is a 3x1 vector, q is a 4x1 vector, outputs the transformed 3x1 vector vec_loc
    R = find_rotation(q)
    return R @ vec

def project(v, on_v):
    norm = np.linalg.norm(on_v)
    if norm != 0.0:
        proj = v @ on_v / norm
        return proj
    else:
        return 0.0

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    velocities = np.empty((0, 3))
    vas = np.empty((0, 1))
    fpas = np.empty((0, 1))
    atts = np.empty((0, 4))
    alphas = np.empty((0, 1))
    va_dots = np.empty((0, 1))
    fpa_dots = np.empty((0, 1))
    n_ps = np.empty((0, 1))
    x_dots = np.empty((0, 3))

    it = 0  # is there a better way to do this?
    for index, row in MergedDf.iterrows():
        # velocity vectors
        velocities = np.append(velocities, [[row['vx'], row['vy'], row['vz']]], axis=0)
        # airspeed v_a
        vas = np.append(vas, [[np.linalg.norm(row['vx':'vz'])]], axis=0)

        # attitude quaternions
        atts = np.append(atts, [[row['q[0]'], row['q[1]'], row['q[2]'], row['q[3]']]], axis=0)
        # Rotation Matrix from body fram to local frame
        R, roll, pitch, yaw = find_rotation(atts[it, :])
        # flight path angles
        # yaw = np.arctan2(R[1,0], 1.0-2*(row['q[0]']**2 + row['q[1]']**2)) #atan2(2.0 * (q.q3 * q.q0 + q.q1 * q.q2) , - 1.0 + 2.0 * (q.q0 * q.q0 + q.q1 * q.q1))
        forward_vector = np.array([1.0 * np.cos(yaw), 1.0 * np.sin(yaw), 0.0])
        if vas[it] != 0:
            fpas = np.append(fpas, [[np.arctan2(- row['vz'], project(velocities[it, :], forward_vector))]], axis=0)  # fpa = arcsin(vz/va) in rad
        else:
            fpas = np.append(fpas, [[0.0]], axis=0)  # fpa = 0 in rad
        # alpha
        alphas = np.append(alphas, [pitch - fpas[it]], axis=0)  # for some reason this gives negative alphas, since for some reason the pitch is always higher than the flight patch angle!
        # va_dots
        body_accel = np.array([row['x'], row['y'], row['z']])
        local_accel = R @ body_accel
        va_dots = np.append(va_dots, [[project(local_accel / 1000.0, velocities[it, :])]], axis=0)
        # fpa_dots
        roll_vec = R @ np.array([[0.0], [-1.0], [0.0]])  # vector pointing along y axis in plane coordinate frame transformed in local frame
        fpa_dot_vec = np.cross(velocities[it, :], np.transpose(roll_vec)[0])
        fpa_dots = np.append(fpa_dots, [[project(local_accel /1000, fpa_dot_vec)]], axis=0)
        # thrust
        u_t = np.array([float(row['control[3]'])]) #TODO: This might change from setup to setup
        n_p = n_0 + u_t * n_slope
        n_ps = np.append(n_ps, [n_p], axis=0)
        # increase counter
        x_dots = np.append(x_dots, [[va_dots[it], fpa_dots[it], it]], axis=0)
        it += 1

    #setup least squares
    # setup least squares for Drag
    y = np.empty((0, 1))
    A = np.empty((0, 2))
    for j in range(fpa_dots.shape[0]):
        t0 = rho * n_ps[j]**2 * D_p**4
        denom1 = np.cos(alphas[j]-thrust_incl)*t0
        denom2 = np.sin(alphas[j]-thrust_incl)*t0
        if denom1 == 0.0 or denom2 == 0.0:
            continue
        L = 0.5 * rho * vas[j] ** 2 * (CL[0] + CL[1] * alphas[j])
        D = 0.5 * rho * vas[j] ** 2 * (CD[0] + CD[1] * alphas[j] + CD[2] * alphas[j] ** 2)
        y_j1 = (m * (va_dots[j]+g*np.sin(fpas[j])) + D)/denom1                 # equation from va_dot
        y_j2 = (m* (fpa_dots[j] + g*np.cos(fpas[j])) - L)/denom2               #don't use, since alphas are very small and introduce error
        A_j = np.array([1, (vas[j]*np.cos(alphas[j]-thrust_incl)/(D_p * n_ps[j]))[0]])
        y = np.append(y, [y_j1], axis=0)
        A = np.append(A, [A_j], axis=0)

    CT = np.linalg.lstsq(A, y, rcond=None)[0]
    print("-------YEY YOU FOUND SOMETHING-------")
    print(f'Thrust Coefficients CT: \n '
          f'  [{CT[0,0]}, {CT[1,0]}]')

    x_dot_pred = np.empty((0, 2))  # v_a_dot, fpa_dot, t
    for j in range(fpa_dots.shape[0]):
        L = 0.5 * rho * vas[j] ** 2 * (CL[0] + CL[1] * alphas[j])
        D = 0.5 * rho * vas[j] ** 2 * (CD[0] + CD[1] * alphas[j] + CD[2] * alphas[j] ** 2)
        T = rho * n_ps[j] ** 2 * D_p ** 4 * (CT[0] + CT[1] * vas[j] * np.cos(alphas[j] - thrust_incl) / (n_ps[j] * D_p))
        pred_va_dot = 1/m*(T*np.cos(alphas[j]-thrust_incl)-D)-g*np.sin(fpas[j])
        pred_fpa_dot = 1/(m)*((T*np.sin(alphas[j]-thrust_incl)+L)-m*g*np.cos(fpas[j]))
        x_dot_pred = np.append(x_dot_pred, [[pred_va_dot[0], pred_fpa_dot[0]]], axis=0)

    fig, axs = plt.subplots(2)

    # plt.figure(1, figsize=(10, 10))
    axs[0].plot(x_dots[0:-1, 2], x_dots[0:-1, 0], 'b', label='Actual va_dot')
    axs[1].plot(x_dots[0:-1, 2], x_dots[0:-1, 1] * 180 / np.pi, 'g', label='Actual fpa_dot')
    axs[0].plot(x_dots[0:-1, 2], x_dot_pred[0:-1, 0], 'r', label='Predicted va_dot')
    axs[1].plot(x_dots[0:-1, 2], x_dot_pred[0:-1, 1] * 180 / np.pi, 'm', label='Predicted fpa_dot')

    axs[0].set_xlabel('Datapoints')
    axs[1].set_xlabel('Datapoints')
    axs[0].set_ylabel('va [m/s]')
    axs[1].set_ylabel('fpa [deg]')
    axs[0].set_title(f'Comparison predicted vs actual derivatives')
    axs[0].grid()
    axs[0].legend(loc='best')
    axs[1].legend(loc='best')
    plt.show()