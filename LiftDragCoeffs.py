import numpy as np
import pandas as pd

#setup
start_min = 3
start_sec = 48
end_min = 4
end_sec = 12
log_name = 'log_01'


#parameters
g =9.81
m = 3.7
rho = 1.25

#setup behind the curtains
start_time = start_min*60+start_sec
end_time = end_min*60+end_sec
AttitudeDF = pd.read_csv(f'./IDFILES/{log_name}_vehicle_attitude_0.csv', index_col=0, usecols=[0, 4, 5, 6, 7])
PosDF = pd.read_csv(f'./IDFILES/{log_name}_vehicle_local_position_0.csv', index_col=0, usecols=[0, 10, 11, 12])
AccelDF = pd.read_csv(f'./IDFILES/{log_name}_sensor_accel_0.csv', index_col=0, usecols=[0, 3, 4, 5])

MergedDf = PosDF.merge(AttitudeDF, left_index=True, right_index=True)
MergedDf = MergedDf.merge(AccelDF, left_index=True, right_index=True)

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

    R[0,0] = aa+bb-cc-dd
    R[0,1] = 2*(bc-ad)
    R[0,2] = 2*(bd+ac)
    R[1,0] = 2*(bc+ad)
    R[1,1] = aa-bb+cc-dd
    R[1,2] = 2*(cd-ab)
    R[2,0] = 2*(bd-ac)
    R[2,1] = 2*(cd+ab)
    R[2,2] = aa-bb-cc+dd
    return R

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
    attitudes = np.empty((0, 4))
    alphas = np.empty((0, 1))
    va_dots = np.empty((0, 1))
    fpa_dots = np.empty((0, 1))

    it = 0 #is there a better way to do this?
    for index, row in MergedDf.iterrows():
        # velocity vectors
        velocities = np.append(velocities, [[row['vx'], row['vy'], row['vz']]], axis=0)
        # airspeed v_a
        vas = np.append(vas, [[np.linalg.norm(row['vx':'vz'])]], axis=0)
        # flight path angles
        if vas[it] != 0:
            fpas = np.append(fpas,[np.arcsin(- row['vz'] / vas[it])], axis=0) #fpa = arcsin(vz/va) in rad
        else:
            fpas = np.append(fpas, [[0.0]], axis=0)  # fpa = 0 in rad
        # attitude quaternions
        attitudes = np.append(attitudes, [[row['q[0]'], row['q[1]'], row['q[2]'], row['q[3]']]], axis=0)
        #alpha
        R = find_rotation(attitudes[it, :])
        pitch = np.arcsin(-R[2, 0])
        alphas = np.append(alphas, [pitch-fpas[it]], axis=0) #for some reason this gives negative alphas, since for some reason the pitch is always higher than the flight patch angle!

        # va_dots
        body_accel = np.array([row['x'], row['y'], row['z']])
        local_accel = R @ body_accel
        va_dots = np.append(va_dots, [[project(local_accel/1000.0, velocities[it, :])]], axis=0)
        # fpa_dots
        roll_vec = R @ np.array([[0.0], [-1.0], [0.0]])  # vector pointing along y axis in plane coordinate frame transformed in local frame
        fpa_dot_vec = np.cross(velocities[it, :], np.transpose(roll_vec)[0])
        fpa_dots = np.append(fpa_dots, [[project(local_accel/1000, fpa_dot_vec)]], axis=0)
        # increase counter
        it +=1
    print(alphas)
    #checked with flight review to be accurate:
    # pitch error <0.01deg, vas, velocities(x,y,z) (x points along runway, y to the right and z down)
    # from pitch = right we can assume attitude = right.



    #setup least squares for Drag
    y = np.empty((0,1))
    A = np.empty((0,3))
    for j in range(MergedDf.shape[0]):
        if vas[j] == 0.0:
            continue
        y_j = 2*m/rho*(va_dots[j]+g*np.sin(fpas[j]))/(vas[j]*vas[j])     
        y = np.append(y, [y_j], axis=0)
        A = np.append(A, [[1.0, alphas[j,0], alphas[j,0]*alphas[j,0]]], axis=0)     

    CD = np.linalg.lstsq(A,y, rcond=None)[0]

    # setup least squares for Lift
    z = np.empty((0, 1))
    B = np.empty((0, 2))
    for j in range(fpa_dots.shape[0]):
        if vas[j] == 0.0:
            continue
        z_j = 2 * m / (rho * vas[j]**2) *(fpa_dots[j]*vas[j] + g * np.cos(fpas[j]))
        z = np.append(z, [z_j], axis=0)
        B = np.append(B, [[1.0, alphas[j, 0]]], axis=0)   

    CL = np.linalg.lstsq(B,z, rcond=None)[0]

    #output
    print("-------YEY YOU FOUND SOMETHING-------")
    print(f'Lift Coefficients: \n  CLs are:    [{CL[0,0]}, {CL[1,0]}]')
    print(f'Drag Coefficients: \n  CDs are:    [{CD[0,0]}, {CD[1, 0]}, {CD[2, 0]}]')

    #TODO:
    #avoid using heading, use measured airspeed instead of calculating it from v_x and v_y
    #what topic and units should I use for the acceleration?
    #compare airspeed log vs what we find here for  airspeed
    # what topics should I use for accel?


# See PyCharm help at https://www.jetbrains.com/help/pycharm/