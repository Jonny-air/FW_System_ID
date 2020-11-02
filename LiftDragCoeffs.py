import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#setup
start_min = 1
start_sec = 43
end_min = 2
end_sec = 3
log_name = 'log_B1'
state_level = 0
show_plots = 1

# start_min = 1
# start_sec = 25.5
# end_min = 1
# end_sec = 45.5
# log_name = 'log_B1'


#parameters
g =9.81
m = 3.204
rho = 1.225

#setup behind the curtains
start_time = start_min*60+start_sec
end_time = end_min*60+end_sec
AttitudeDF = pd.read_csv(f'./IDFILES/{log_name}_vehicle_attitude_0.csv', index_col=0, usecols=[0, 4, 5, 6, 7])
PosDF = pd.read_csv(f'./IDFILES/{log_name}_vehicle_local_position_0.csv', index_col=0, usecols=[0, 10, 11, 12])
AccelDF = pd.read_csv(f'./IDFILES/{log_name}_sensor_accel_0.csv', index_col=0, usecols=[0, 3, 4, 5])

MergedDf = pd.merge_asof(PosDF, AttitudeDF, left_index=True, right_index=True)
MergedDf = pd.merge_asof(MergedDf, AccelDF, left_index=True, right_index=True)

#only take lines we want
MergedDf = MergedDf.query(f'timestamp >= {start_time*1.0E6} and timestamp <= {end_time*1.0E6}')

#Merged.Df.iloc[i] returns ith row of df
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
    x_dots = np.empty((0, 3))

    it = 0 #is there a better way to do this?
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
        #yaw = np.arctan2(R[1,0], 1.0-2*(row['q[0]']**2 + row['q[1]']**2)) #atan2(2.0 * (q.q3 * q.q0 + q.q1 * q.q2) , - 1.0 + 2.0 * (q.q0 * q.q0 + q.q1 * q.q1))

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
            local_accel = local_accel/1000
        else:
            local_accel[2] += g
        va_dots = np.append(va_dots, [[project(local_accel, velocities[it, :])]], axis=0)
        # fpa_dots
        fpa_dot_vec = - np.cross(velocities[it, :], np.transpose(right_vector))
        fpa_dots = np.append(fpa_dots, [[project(local_accel, fpa_dot_vec)]], axis=0)
        # increase counter
        x_dots = np.append(x_dots, [[va_dots[it], fpa_dots[it], it]], axis=0)
        it +=1

    # state level vs real
    # y is forward, x is left, z is down
    # roll, pitch, yaw are right

    #setup least squares for Drag
    y = np.empty((0,1))
    A = np.empty((0,3))
    for j in range(MergedDf.shape[0]):
        if vas[j] == 0.0:
            continue
        y_j = -2*m/rho*(va_dots[j]+g*np.sin(fpas[j]))/(vas[j]*vas[j])
        y = np.append(y, [y_j], axis=0)
        A = np.append(A, [[1.0, alphas[j,0], alphas[j,0]*alphas[j,0]]], axis=0)     

    CD, res1 = np.linalg.lstsq(A,y, rcond=None)[0:2]

    # setup least squares for Lift
    z = np.empty((0, 1))
    B = np.empty((0, 2))
    data_lift = np.empty((0, 2))
    for j in range(fpa_dots.shape[0]):
        if vas[j] == 0.0:
            continue
        z_j = 2 * m / (rho * vas[j]**2) *(fpa_dots[j] + g * np.cos(fpas[j]))
        z = np.append(z, [z_j], axis=0)
        B = np.append(B, [[1.0, alphas[j, 0]]], axis=0)

    CL, res2 = np.linalg.lstsq(B,z, rcond=None)[0:2]
    #output
    print("-------YEY YOU FOUND SOMETHING-------")
    print(f'Drag Coefficients CD: \n  [{CD[0,0]}, {CD[1, 0]}, {CD[2, 0]}]')
    print(f'Lift Coefficients CL: \n  [{CL[0,0]}, {CL[1, 0]}]')

    #show fit, for every timestep calculate actual xdot and predicted xdot
    #actual xdots
    if show_plots:
        x_dot_pred = np.empty((0, 3)) #v_a_dot, fpa_dot, t
        for j in range(fpa_dots.shape[0]):
            L = 0.5 * rho * vas[j]**2 * (CL[0, 0] + CL[1, 0] * alphas[j, 0])
            D = 0.5 * rho * vas[j]**2 * (CD[0, 0] + CD[1, 0] * alphas[j, 0] + CD[2, 0] * alphas[j, 0]**2)
            pred_va_dot  = 1/m * -D - g * np.sin(fpas[j])
            pred_fpa_dot = 1/m *  L - g * np.cos(fpas[j])
            x_dot_pred = np.append(x_dot_pred, [[pred_va_dot[0], pred_fpa_dot[0], j]], axis = 0)

        fig, axs = plt.subplots(2, figsize=(10, 10))

        axs[0].plot(x_dots[0:-1, 2], x_dots[0:-1, 0], 'b', label='Actual va_dot')
        axs[1].plot(x_dots[0:-1, 2], x_dots[0:-1, 1], 'g', label='Actual fpa_dot*va')
        axs[0].plot(x_dots[0:-1, 2], x_dot_pred[0:-1, 0], 'r', label='Predicted va_dot')
        axs[1].plot(x_dots[0:-1, 2], x_dot_pred[0:-1, 1], 'm', label='Predicted fpa_dot*va')


        axs[0].set_xlabel('Datapoints')
        axs[1].set_xlabel('Datapoints')
        axs[0].set_ylabel('va_dot [m/s2]')
        axs[1].set_ylabel('fpa_dot*va [m/s2]')
        axs[0].set_title(f'Comparison predicted vs actual derivatives')
        axs[0].grid()
        axs[1].grid()
        axs[0].legend(loc='best')
        axs[1].legend(loc='best')
        plt.show()

        alpha_start = -5 * np.pi / 180
        alpha_end = 10 * np.pi / 180
        alpha_step = 0.1 * np.pi / 180
        L = np.empty((0, 2))
        D = np.empty((0, 2))
        for alpha in np.arange(alpha_start, alpha_end, alpha_step):
            L = np.append(L, [[(CL[0] + CL[1] * alpha), alpha]], axis=0)
            D = np.append(D, [[(CD[0] + CD[1] * alpha + CD[2] * alpha ** 2), alpha]], axis=0)



        # Lift/Drag
        fig, axs = plt.subplots(2, figsize=(10, 10))
        ax1 = axs[0]
        ax2 = axs[1]

        ax1.plot(L[0:-1, 1] * 180 / np.pi, L[0:-1, 0], 'b', label='Lift')
        ax1.plot((alphas[0:-1])*180 / np.pi, z[0:-1, 0], '-x', label='Lift Datapoints')
        plt.figtext(.7, .75, f"Residual = {round(res2[0],5)}")
        ax2.plot(D[0:-1, 1] * 180 / np.pi, D[0:-1, 0], 'r', label='Drag')
        ax2.plot(D[0:-1, 1] * 180 / np.pi, L[0:-1, 0]/D[0:-1, 0], 'g', label='Lift/Drag')
        ax2.plot((alphas[0:-1]) * 180 / np.pi, y[0:-1, 0], '-x', label='Drag Datapoints')
        plt.figtext(.7, .26, f"Residual = {round(res1[0],5)}")

        ax1.set_xlabel('Angle of Attack in Degrees')
        ax2.set_xlabel('Angle of Attack in Degrees')
        ax1.set_ylabel('Lift Coefficients [m^2]')
        ax2.set_ylabel('Drag Coefficients [m^2]')
        plt.title(f'Lift and Drag Coefficients (incl. surface area) depending on AoA')
        ax1.legend(loc='best')
        ax2.legend(loc='best')
        ax1.grid()
        ax2.grid()
        plt.show()
#checked with flight review to be accurate:
# pitch, yaw error <0.01deg, vas, velocities(x,y,z) (x points along runway, y to the right and z down)
# from pitch = right we can assume attitude = right.