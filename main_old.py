import numpy as np
import csv

AttitudeCSV = open('./IDFILES/log_01_vehicle_attitude_0.csv', mode='r')
PosCSV = open('./IDFILES/log_01_vehicle_local_position_0.csv', mode='r')
AccelCSV = open('./IDFILES/log_01_sensor_accel_0.csv', mode='r')

#setup


#parameters
g =9.81
m = 3.7
rho = 1.25

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
    vec = np.array([[1, 0, 0], [0,1,0], [2, 1, 1]])
    q = np.array([[0.997582, 0.0213816, 0.0261228, -0.0608291], [0.997582, 0.0213816, 0.0261228, -0.0608291],
                  [0.997582, 0.0213816, 0.0261228, -0.0608291]])

    v_l = np.empty((0,3))
    for j in range(vec.shape[0]):
        v_l = np.append(v_l, [transform_to_local(vec[j], q[j])], axis=0)
        project(v_l[j], vec[j])

    #velocities:::
    POSreader = csv.reader(PosCSV, delimiter=',', quotechar='|')
    next(POSreader) #skip first line
    velocities = np.empty((0,3))
    vas = np.empty((0,1))
    fpas = np.empty((0, 1))
    for row in POSreader:
        velocities = np.append(velocities, [[float(row[10]), float(row[11]), float(row[12])]], axis=0)
        va = np.linalg.norm(row[10:13])
        vas = np.append(vas, [[va]], axis=0)
        if va != 0:
            x = -float(row[12]) / va
            fpas = np.append(fpas,[[np.arcsin(x)]], axis=0) #fpa = arccos(vz/va) in rad
        else:
            fpas = np.append(fpas, [[0.0]], axis=0)  # fpa = 0 in rad

    #Attitudes:::
    ATTreader = csv.reader(AttitudeCSV, delimiter=',', quotechar='|')
    next(ATTreader)# skip line, we want 4,5,6,7 = q0, q1, q2, q3 attitude quaternion in local inertial frame
    attitudes = np.empty((0, 4))
    for row in ATTreader:
        attitude = np.array([float(row[4]), float(row[5]), float(row[6]), float(row[7])])
        attitudes = np.append(attitudes, [attitude], axis=0)

    #accelerations:::
    ACCELreader = csv.reader(AccelCSV, delimiter=',', quotechar='|')
    next(ACCELreader) # skip line, we want 3,4,5 = x,y,z body NED
    alphas = np.empty((0, 1))
    va_dots = np.empty((0, 1))
    fpa_dots = np.empty((0, 1))
    index = 0
    for row in ACCELreader:
        body_accel = np.array([float(row[3]), float(row[4]), float(row[5])])
        R = find_rotation(attitudes[index,:])
        local_accel = R @ body_accel
        #pitch: asin(2.0 * (q.q2 * q.q0 - q.q3 * q.q1))
        pitch = np.arcsin(-R[2, 0])
        #roll: atan2(2.0 * (q.q3 * q.q2 + q.q0 * q.q1) , 1.0 - 2.0 * (q.q1 * q.q1 + q.q2 * q.q2))
        #roll = np.arctan2(R[2,1] , 1.0 - 2.0 * (attitudes[index,1]*attitudes[index,1]+ attitudes[index,2]*attitudes[index,2]))

        #alpha
        alpha = float(pitch - fpas[j]) #TODO fix this. dont go by rows but rather by timestamps
        alphas = np.append(alphas, [[alpha]], axis=0)

        #va_dot
        va_dot = project(local_accel/100.0, velocities[j,:]) #TODO fix this. dont go by rows but rather by timestamps
        va_dots = np.append(va_dots, [[va_dot]], axis=0)

        #fpa_dot
        fpa_dot_vec = R @ np.array([[0.0], [0.0], [-1.0]]) #vector pointing up in plane coordinate frame transformed in local frame
        fpa_dot = project(local_accel/100, fpa_dot_vec)
        fpa_dots = np.append(fpa_dots, [fpa_dot], axis=0)

        index += 1

    #setup least squares for Drag
    y = np.empty((0,1))
    A = np.empty((0,3))
    for j in range(vas.shape[0]):
        if vas[j] == 0.0:
            continue
        y_j = 2*m/rho*(va_dots[j]+g*np.sin(fpas[j]))/(vas[j]*vas[j])    #TODO fix this. dont go by rows but rather by timestamps
        y = np.append(y, [y_j], axis=0)
        A = np.append(A, [[1.0, alphas[j,0], alphas[j,0]*alphas[j,0]]], axis=0)    #TODO fix this. dont go by rows but rather by timestamps

    CD = np.linalg.lstsq(A,y, rcond=None)[0]

    # setup least squares for Lift
    z = np.empty((0, 1))
    B = np.empty((0, 2))
    for j in range(fpa_dots.shape[0]):
        if vas[int(round(j / 2))] == 0.0:
            continue
        z_j = 2 * m / (rho * vas[int(round(j/2))]**2) *(fpa_dots[j]*vas[int(round(j/2))] + g * np.cos(fpas[int(round(j / 2))]))   # TODO fix this. dont go by rows but rather by timestamps
        z = np.append(z, [z_j], axis=0)
        B = np.append(B, [[1.0, alphas[j, 0]]], axis=0)  # TODO fix this. dont go by rows but rather by timestamps

    CL = np.linalg.lstsq(B,z, rcond=None)[0]

    #output
    print("-------YEY YOU FOUND SOMETHING-------")
    print(f'Lift Coefficients: \n  C_l0 is:    {CL[0,0]} \n  C_l_alpha is:    {CL[1,0]}')
    print(f'Drag Coefficients: \n  C_d0 is:    {CD[0, 0]} \n  C_d_alpha is:    {CD[1, 0]}\n  C_d_alpha2 is:    {CD[2, 0]}')

    #TODO:
    #avoid using heading, use measured airspeed instead of calculating it from v_x and v_y
    #what topic and units should I use for the acceleration?
    #compare airspeed log vs what we find here for  airspeed
    # what topics should I use for accel?


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
