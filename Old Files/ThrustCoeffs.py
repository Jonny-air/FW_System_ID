import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import median_filter

#setup
start_min = 3
start_sec = 24
end_min = 4
end_sec = 5
log_name = 'log_BT2'
state_level = 0
show_plots = 0
filter_alpha = True



#parameters
g =9.80
m = 3.204
rho = 1.225
n_0 = 15.00/(2*np.pi)   #rotps
n_slope = 985/(2*np.pi) #rotps
thrust_incl = 0.0 #rad
D_p = 0.28

# CD = np.array(
#     [0.026871082831117384, 0.18575809516924544, 0.6866415654031391]
# )
# CL = np.array(
#     [0.10678183352119403, 1.8826298036731293]
# )
CD = np.array([0.02369842, 0.07307902, 0.68664157])
CL = np.array([0.11768223, 1.51342619])

#setup behind the curtains
start_time = start_min*60+start_sec
end_time = end_min*60+end_sec
AttitudeDF = pd.read_csv(f'./csv/{log_name}_vehicle_attitude_0.csv', index_col=0, usecols=[0, 4, 5, 6, 7])
PosDF = pd.read_csv(f'./csv/{log_name}_vehicle_local_position_0.csv', index_col=0, usecols=[0, 10, 11, 12])
AccelDF = pd.read_csv(f'./csv/{log_name}_sensor_accel_0.csv', index_col=0, usecols=[0, 3, 4, 5])
ActuatorDF = pd.read_csv(f'./csv/{log_name}_actuator_controls_0_0.csv', index_col=0, usecols=[0, 5])

MergedDf = pd.merge_asof(PosDF, AttitudeDF, left_index=True, right_index=True)
MergedDf = pd.merge_asof(MergedDf, AccelDF, left_index=True, right_index=True)
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
    rolls = np.empty((0, 1))
    alphas = np.empty((0, 1))
    va_dots = np.empty((0, 1))
    fpa_dots = np.empty((0, 1))
    n_ps = np.empty((0, 1))
    x_dots = np.empty((0, 3))

    it = 0  # is there a better way to do this?
    for index, row in MergedDf.iterrows():
        # velocity vectors
        velocities = np.append(velocities, [[row['vx'], row['vy'], row['vz']]], axis=0)
        # attitude quaternions
        atts = np.append(atts, [[row['q[0]'], row['q[1]'], row['q[2]'], row['q[3]']]], axis=0)
        # Rotation Matrix from body fram to local frame
        R, roll, pitch, yaw = find_rotation(atts[it, :])
        rolls = np.append(rolls, [[roll]], axis=0)
        forward_vector = np.array([1.0 * np.cos(yaw), 1.0 * np.sin(yaw), 0.0])
        right_vector = np.array([-1.0 * np.sin(yaw), 1.0 * np.cos(yaw), 0.0])
        # airspeed v_a
        vas = np.append(vas, [[project(velocities[it, :], forward_vector)]], axis=0)
        # flight path angles
        # yaw = np.arctan2(R[1,0], 1.0-2*(row['q[0]']**2 + row['q[1]']**2)) #atan2(2.0 * (q.q3 * q.q0 + q.q1 * q.q2) , - 1.0 + 2.0 * (q.q0 * q.q0 + q.q1 * q.q1))

        if vas[it] != 0:
            fpas = np.append(fpas, [np.arctan2(- row['vz'], vas[it])], axis=0)  # fpa = arcsin(vz/va) in rad
        else:
            fpas = np.append(fpas, [[0.0]], axis=0)  # fpa = 0 in rad
        # alpha
        alphas = np.append(alphas, [pitch - fpas[it]], axis=0)
        # va_dots
        body_accel = np.array([row['x'], row['y'], row['z']])
        local_accel = R @ body_accel
        if state_level:
            local_accel = local_accel / 1000
        else:
            local_accel[2] += g
        va_dots = np.append(va_dots, [[project(local_accel, velocities[it, :])]], axis=0)
        # fpa_dots
        fpa_dot_vec = - np.cross(velocities[it, :], np.transpose(right_vector))
        fpa_dots = np.append(fpa_dots, [[project(local_accel, fpa_dot_vec)]], axis=0)
        # thrust
        u_t = float(row['control[3]']) #TODO: This might change from setup to setup
        n_p = n_0 + u_t * n_slope
        n_ps = np.append(n_ps, [[n_p]], axis=0)
        # increase counter
        it += 1
    print(fpa_dots)

    if filter_alpha:
        # median filter
        median_filter(alphas, size=(7, 1), output=alphas)

    #setup least squares
    def Thrust_func(CT):
        T = rho*n_ps**2*D_p**4*( CT[0] + CT[1] * vas*np.cos(alphas-thrust_incl)/(D_p * n_ps))
        D = 0.5 * rho * vas ** 2 * (CD[0] + CD[1] * alphas + CD[2] * alphas ** 2)
        err = -va_dots + 1 / m *(T*np.cos(alphas - thrust_incl) -D) - g * np.sin(fpas)
        #weigh datapoints around normal operation point heavier
        err *= np.cos((n_ps-100)*0.0125) #a difference of 80 equals to 0.7 weight
        return err.transpose()[0]

    D = 0.5 * rho * vas ** 2 * (CD[0] + CD[1] * alphas + CD[2] * alphas ** 2)
    t0 = rho * n_ps ** 2 * D_p ** 4
    denom1 = np.cos(alphas-thrust_incl)*t0
    datapoints = (m * (va_dots  + g*np.sin(fpas)) + D) / denom1

    start_CT = np.array([0.20245296276934735, -0.2993443648040376])
    sol = least_squares(Thrust_func, start_CT, loss="cauchy", f_scale=1.0, bounds=((-3, -3), (3, 3)))
    CT = sol.x
    res3 = sol.cost



    # y = np.empty((0, 1))
    # A = np.empty((0, 2))
    # datapoints = np.empty((0, 3))
    # for j in range(fpa_dots.shape[0]):
    #     t0 = rho * n_ps[j]**2 * D_p**4
    #     denom1 = np.cos(alphas[j]-thrust_incl)*t0
    #     # denom2 = np.sin(alphas[j]-thrust_incl)*t0
    #     if n_ps[j] < 20: #discard low prop speeds where drag inaccurcy is too high
    #         continue
    #     # L = 0.5 * rho * vas[j] ** 2 * (CL[0] + CL[1] * alphas[j])
    #     D = 0.5 * rho * vas[j] ** 2 * (CD[0] + CD[1] * alphas[j] + CD[2] * alphas[j] ** 2)
    #     y_j1 = (m * (va_dots[j]  + g*np.sin(fpas[j])) + D) / denom1                 # equation from va_dot
    #     # y_j2 = (m * (fpa_dots[j] + g*np.cos(fpas[j])) - L) / denom2               #don't use, since alphas are very small and introduce error
    #     A_j = np.array([1.0, (vas[j]*np.cos(alphas[j]-thrust_incl)/(D_p * n_ps[j]))[0]])
    #     y = np.append(y, [y_j1], axis=0)
    #     datapoints = np.append(datapoints, [[y_j1[0], n_ps[j], vas[j]]], axis=0)
    #     A = np.append(A, [A_j], axis=0)
    #
    # CT, res3 = np.linalg.lstsq(A, y, rcond=None)[0:2]
    print("-------YEY YOU FOUND SOMETHING-------")
    print(f'Thrust Coefficients CT: \n '
          f'  [{CT[0]}, {CT[1]}]')

    L = 0.5 * rho * vas ** 2 * (CL[0] + CL[1] * alphas)
    D = 0.5 * rho * vas ** 2 * (CD[0] + CD[1] * alphas + CD[2] * alphas ** 2)
    T = rho * n_ps ** 2 * D_p ** 4 * (CT[0] + CT[1] * vas * np.cos(alphas - thrust_incl) / (n_ps * D_p))

    pred_va_dot =  1/m* (T*np.cos(alphas-thrust_incl) -D)   -g*np.sin(fpas)
    pred_fpa_dot = 1/m *(T*np.sin(alphas-thrust_incl) +L)*np.cos(rolls)  -g*np.cos(fpas)


    fig, axs = plt.subplots(2)
    t = range(1, fpas.shape[0], 1)

    # plt.figure(1, figsize=(10, 10))
    axs[0].plot(t, va_dots[0:-1], 'b', label='Actual va_dot')
    axs[1].plot(t, fpa_dots[0:-1], 'g', label='Actual fpa_dot')
    axs[0].plot(t, pred_va_dot[0:-1], 'r', label='Predicted va_dot')
    axs[1].plot(t, pred_fpa_dot[0:-1], 'm', label='Predicted fpa_dot')

    axs[0].set_xlabel('Datapoints')
    axs[1].set_xlabel('Datapoints')
    axs[0].set_ylabel('va [m/s]')
    axs[1].set_ylabel('fpa [deg]')
    axs[0].set_title(f'Comparison predicted vs actual derivatives')
    axs[0].grid()
    axs[0].legend(loc='best')
    axs[1].legend(loc='best')
    plt.show()

    #Thrust Coefficients Plot
    n_p = np.arange(n_0, n_0 + n_slope, 15)
    inflow = np.arange(0, 30, 1)
    n_p, inflow = np.meshgrid(n_p, inflow)
    T = (CT[0] + CT[1] * inflow / (n_p *D_p))
    T = (CT[0] + CT[1] * inflow / (n_p *D_p))
    data_inflows = vas*np.cos(alphas-thrust_incl)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(n_p, inflow, T, color='green')
    ax.scatter(n_ps[0:-1], data_inflows[0:-1], datapoints[0:-1], zdir ='z', s =10)
    ax.set_xlabel('n_p in rot/sec')
    ax.set_ylabel('inflow va*cos(alpha-thrust_incl) in m/s')
    ax.set_zlabel('Combined Thrust coeficient [1]')
    ax.set_zlim(-0.3,0.1)
    plt.show()