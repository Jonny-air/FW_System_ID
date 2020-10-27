import numpy as np
import csv

#parameters
g =9.81
m = 3.7
rho = 1.25
n_0 = 15.03 #radps
n_slope = 1157.4
thrust_incl = 0.0 #rad
D_p = 0.28
CD = np.array([78.76906, 59.33149, -307.0041])
CL = np.array([31.10244, 62.75034])

ActuatorCSV = open('./IDFILES/log_01_actuator_controls_0_0.csv', mode='r')
AttitudeCSV = open('./IDFILES/log_01_vehicle_attitude_0.csv', mode='r')
PosCSV = open('./IDFILES/log_01_vehicle_local_position_0.csv', mode='r')
VelCSV = open('./IDFILES/log_01_airspeed_0.csv', mode='r')
AccelCSV = open('./IDFILES/log_01_sensor_accel_0.csv', mode='r')

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
        alpha = float(pitch - fpas[int(round(index/2))]) #TODO fix this. dont go by rows but rather by timestamps
        alphas = np.append(alphas, [[alpha]], axis=0)

        #va_dot
        va_dot = project(local_accel/100.0, velocities[int(round(index/2)),:]) #TODO fix this. dont go by rows but rather by timestamps
        va_dots = np.append(va_dots, [[va_dot]], axis=0)

        #fpa_dot
        fpa_dot_vec = R @ np.array([[0.0], [0.0], [-1.0]]) #vector pointing up in plane coordinate frame transformed in local frame
        fpa_dot = project(local_accel/100, fpa_dot_vec)
        fpa_dots = np.append(fpa_dots, [fpa_dot], axis=0)

        index += 1

    #Thrust:::
    ACTreader = csv.reader(ActuatorCSV, delimiter=',', quotechar='|')
    next(ACTreader)  # skip line, we want 4,5,6,7 = q0, q1, q2, q3 attitude quaternion in local inertial frame
    n_ps = np.empty((0, 1))
    for row in ACTreader:
        u_t = np.array([float(row[5])])
        n_p = n_0 + u_t*n_slope
        n_ps = np.append(n_ps, [n_p], axis=0)

    #setup least squares
    # setup least squares for Drag
    y = np.empty((0, 1))
    A = np.empty((0, 2))
    for j in range(fpa_dots.shape[0]):
        t0 = rho * n_ps[j]**2 * D_p ** 4
        denom1 = np.cos(alphas[j]-thrust_incl)*t0
        denom2 = np.sin(alphas[j]-thrust_incl)*t0
        if denom1 == 0.0 or denom2 == 0.0:
            continue
        D = 0.5*rho*vas[int(round(j / 2))]**2*(CD[0] + CD[1]*alphas[j] + CD[2]*alphas[j]**2)
        L = 0.5*rho*vas[int(round(j / 2))]**2*(CL[0] + CL[1]*alphas[j])
        y_j1 = (m * va_dots[j]*g*np.sin(fpas[int(round(j / 2))]) + D)/ denom1 # equation vrom va_dot TODO fix this. dont go by rows but rather by timestamps
        y_j2 = (m*(fpa_dots[j]*vas[int(round(j / 2))] + g*np.cos(fpas[int(round(j / 2))])) - L)/denom2
        A_j = np.array([1, (vas[int(round(j / 2))]*np.cos(alphas[j]-thrust_incl)/(D_p * n_ps[j]))[0]])
        y = np.append(y, [y_j1, y_j2], axis=0)
        A = np.append(A, [A_j, A_j], axis=0)  # TODO fix this. dont go by rows but rather by timestamps

    CT = np.linalg.lstsq(A, y, rcond=None)[0]
    print("-------YEY YOU FOUND SOMETHING-------")
    print(f'Thrust Coefficients: \n  C_t0 is:    {CT[0,0]} \n  C_t1 is:    {CT[1,0]}')