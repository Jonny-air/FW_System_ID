import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.optimize import fmin
import math
import Calc_Coefficients
from load_model import get_params
from Lift_Calculation import get_lift
from Drag_Calculation import get_drag
from Thrust_Calculation import get_thrust

#Setup Coefficient Calculation
model_name = "Believer_201112_true"
verbose = False
show_plots = False
fix_cd2 = False

# Environment Parameters
g =9.80
rho = 1.225
env = g, rho

#other data for lookup not contained in model
max_rps = 1000
max_thrust = 1
min_thrust = 0
max_pitch = 30*np.pi/180
min_pitch = -30*np.pi/180
max_bank = 0.6

# Techpod
# CD = np.array(
#     [0.0603, 0.34728666767791942, 0.2599156540313911]
# )
# CL = np.array(
#     [0.52154, 2.8514436703100506]
# )
# CT = np.array(
#     [0.1152, -0.1326]
# )
# Believer
# CD = np.array([0.023698424821326284, 0.07307901552413693, 0.6866415654031391])
# CL = np.array([0.11768223035003972, 1.513426191916197])
# CT = np.array(
#     # [0.183943033572054, -0.21233020186717652]
#     [0.13286325381266204, -0.09561982458759365]
#     [0.13128021890289707, -0.09461411581945008]
# )

CL, CD = Calc_Coefficients.calc_lift_drag(model_name, fix_cd2, verbose, show_plots, env)
CT = Calc_Coefficients.calc_thrust(model_name, CL, CD, verbose, show_plots, env)
[m, n_0, n_slope, thrust_incl, D_p, actuator_control_ch, state_level] = get_params(model_name, verbose)


def state_func(va, fpa, roll, alpha, u_t):
    n_p = n_slope * u_t + n_0
    # find current error
    T = get_thrust(CT, va, alpha, n_p, thrust_incl, D_p)
    L = get_lift(CL,va,alpha)
    D = get_drag(CD,va,alpha)

    x1 = 1 / m * (T * np.cos(alpha - thrust_incl) - D) - g * np.sin(fpa)
    x2 = 1 / (m * va) * ((T * np.sin(alpha - thrust_incl) + L) * np.cos(roll) - m * g * np.cos(fpa))
    return([x1, x2])

def setup_solver(x, case, input):
    if case == 0:
        alpha, u_t, roll = input
        va, fpa = x
        return state_func(va, fpa, roll, alpha, u_t)
    if case == 2 or case == 3:
        alpha, fpa, roll = input
        va, u_t = x
        return state_func(va, fpa, roll, alpha, u_t)
    if case == 6:
        u_t, fpa, roll = input
        va, pitch = x
        alpha = pitch - fpa
        return state_func(va, fpa, roll, alpha, u_t)
    elif case == 1 or case > 3:
        va, fpa, roll = input
        u_t, pitch = x
        alpha = pitch - fpa
        return state_func(va, fpa, roll, alpha, u_t)

def populate_table(L, roll):
    # min sink set alpha to maxcl3cd2, throttle=0, solve for fpa and va -> case 0
    case = 0
    input = np.append(L[[0, 3], case], [roll])  # alpha, u_t, roll
    start_x = L[[1, 2], case]  # va, fpa
    roots = least_squares(setup_solver, start_x, args=(case, input), bounds=((5, -0.6),(23,0.6)))

    if np.isclose(setup_solver(roots.x, case, input), [0.0, 0.0]).all():
        va, fpa = roots.x[0:2]
        pitch = L[0,case] + fpa
        L[1,case] = va
        L[2,case] = fpa
        L[4,case] = pitch
    else:
        print(f"[ERROR] Case {case} with roll {roll} did not converge")

    case=1
    L[1, case] = L[1, 0] #take velocity from case 1
    L[2, case] = L[2, 0] + (1*np.pi/180) #add some fpa to it
    input = np.append(L[[1, 2], case], [roll])  # va, fpa, roll
    start_x = L[[3, 4], case]  # u_t, pitch
    roots = least_squares(setup_solver, start_x, args=(case, input), bounds=((0, -0.6),(1,0.6))).x
    if np.isclose(setup_solver(roots, case, input), [0.0, 0.0]).all():
        u_t, pitch = roots[0:2]
        L[3, case] = u_t
        L[4, case] = pitch
    else:
        print(f"[ERROR] Case {case} with roll {roll} did not converge")

    for case in range(2,4,1):
        input = np.append(L[[0, 2], case], [roll]) # alpha, fpa, roll
        start_x = L[[1, 3], case]  # va, u_t
        roots = least_squares(setup_solver, start_x, args=(case, input), bounds=((5, 0),(30, 1))).x
        if np.isclose(setup_solver(roots, case, input), [0.0, 0.0]).all():
            va, u_t = roots[0:2]
            pitch = L[0, case] + L[2, case]
            L[4, case] = pitch
            L[1, case] = va
            L[3, case] = u_t
        else:
            print(f"[ERROR] Case {case} with roll {roll} did not converge")

    case = 6
    input = np.append(L[[3, 2], case], [roll])  # u_t, fpa, roll
    start_x = L[[1, 4], case]  # va, pitch
    roots = least_squares(setup_solver, start_x, args=(case, input), bounds=((0.1, -0.5), (30, 0.5))).x
    if np.isclose(setup_solver(roots, case, input), [0.0, 0.0]).all():
        va, pitch = roots[0:2]
        L[1, case] = va
        L[4, case] = pitch
    else:
        print(f"[ERROR] Case {case} with roll {roll} did not converge")

    #update enries for max_sink_p
    L[1, 5] = L[1, 6]
    L[2, 5] = L[2, 6]-(1*np.pi/180)

    for case in range(4,6,1):
        input = np.append(L[[1, 2], case], [roll])  # va, fpa, roll
        start_x = L[[3, 4], case]  # u_t, pitch
        roots = least_squares(setup_solver, start_x, args=(case, input), bounds=((0.0, -0.5),(1,0.5))).x
        if np.isclose(setup_solver(roots, case, input), [0.0, 0.0]).all():
            u_t, pitch = roots[0:2]
            L[3, case] = u_t
            L[4, case] = pitch
        else:
            print(f"[ERROR] Case {case} with roll {roll} did not converge")

    return L

def main():
    # set up tables
    #find alpha for Lift/Drag max
    a = -CL[1]*CD[2]
    b = -2*CL[0]*CD[2]
    c = CL[1]*CD[0] -CL[0]*CD[1]
    root = math.sqrt(b*b - 4*a*c)
    x1 = 1/(2.0*a)*((-b) + root) #generally we want to take the high alpha case for which we are slow, so x2 instead of this
    ldmax = 1/(2.0*a)*((-b) - root)

    # for min sink find cl3/cd2 max
    def cl3cd2(alpha):
        func = (CD[0] + CD[1] * alpha + CD[2] * alpha ** 2)/(CL[0] + CL[1] * alpha)**(3/2)
        return func
    maxcl3cd2 = fmin(cl3cd2, 0.0)
    if verbose:
        print(f"max lift/drag at {ldmax*180/np.pi}")
        print(f"max cl3cd2 at {maxcl3cd2*180/np.pi}")
        L = np.empty((0,2))
        D = np.empty((0, 2))
        for alpha in np.arange(-10*np.pi/180, 30*np.pi/180, 0.01):
            L = np.append(L, [[(CL[0] + CL[1] * alpha), alpha]], axis=0)
            D = np.append(D, [[(CD[0] + CD[1] * alpha + CD[2] * alpha**2), alpha]], axis=0)
        Y1 = (CL[0] + CL[1] * x1) / (CD[0] + CD[1] * x1 + CD[2] * ldmax ** 2)
        Y2 = (CL[0] + CL[1] * ldmax) / (CD[0] + CD[1] * ldmax + CD[2] * ldmax ** 2)
        Y3 = (CL[0] + CL[1] * maxcl3cd2) / (CD[0] + CD[1] * maxcl3cd2 + CD[2] * maxcl3cd2 ** 2)

        plt.figure(1, figsize=(10, 10))
        plt.plot(L[0:-1, 1] * 180 / np.pi, L[0:-1, 0], 'b', label='Lift')
        plt.plot(D[0:-1, 1] * 180 / np.pi, D[0:-1, 0], 'r', label='Drag')
        plt.plot(L[0:-1, 1] * 180 / np.pi, L[0:-1, 0]/D[0:-1, 0], 'g', label="Lift/Drag Ratio")
        plt.plot(x1*180/np.pi, Y1, '-o', Label="x1")
        plt.plot(maxcl3cd2*180/np.pi, Y2, '-o', Label="maxcl3cd2")
        plt.plot(ldmax*180/np.pi, Y2, '-o', Label="ldmax")

        plt.legend(loc='best')
        plt.title(f'Lift, Drag and Ratio of Coefficients')
        plt.grid()
        plt.show()

    L1 = np.zeros((5, 7))
    L2 = np.zeros((5,7))
    # table 0. correspond to not set values, everything else is needed or used as starting conditions (check mxcl3cd2 for reasonable value and set maxcr high enough)
    #        minsink         _p    max climb      crs     maxcr      maxsink_p     maxsink
    L1[0] = [  maxcl3cd2,     ldmax,       ldmax,    0.034906,    0.002,     ldmax,         0.]  # 0 alpha
    L1[1] = [  13.50000, 11.790514, 13.50000, 14.00000, 26.00000,  26.00000,   26.00000]  # 1 airspeed
    L1[2] = [ -0.130899, -0.095993, 0.349065, 0.000000, 0.000000, -0.174532,  -0.209439]  # 2 fpa
    L1[3] = [   0.0,       0.353,     0.89,      0.55,    0.68,       0.3,        0.0]  # 3 throttle for min sink and max sink
    L1[4] = [   0.0,        0.01,     0.5,        0.1,     0.1,       -0.2,      -0.2]  # 4 pitch starting points

    L2[0] = [  maxcl3cd2,     ldmax,       ldmax,    0.034906,    0.002,     ldmax,         0.]  # 0 alpha
    L2[1] = [  13.50000, 11.790514, 13.50000, 14.00000, 26.00000,  26.00000,   26.00000]  # 1 airspeed
    L2[2] = [ -0.130899, -0.095993, 0.027925, 0.000000, 0.000000, -0.174532,  -0.209439]  # 2 fpa
    L2[3] = [   0.0,       0.353,     0.89,      0.55,    0.68,       0.3,        0.0]  # 3 throttle for min sink and max sink
    L2[4] = [   0.0,        0.01,     0.5,        0.1,     0.1,       -0.2,      -0.2]  # 4 pitch starting points

    roll = 0.0
    L1 = populate_table(L1, roll)

    input_vs_roll = np.zeros((0, 3))  # throttle, pitch, roll, for max cruise
    for i in np.linspace(0.0, max_bank, 10):
        roll = i
        L = L2
        L = populate_table(L, roll)
        print(f"populated for roll {i}")
        input_vs_roll = np.append(input_vs_roll, [[L[3, 4], L[4, 4], roll]], axis=0)

    def func_est(start_point, end_point, roll, coeff):
        help_a = (end_point-start_point)/(2*max_bank*max_bank)*coeff
        return help_a*roll*roll + start_point

    fig, axs = plt.subplots(2, figsize=(10,10))
    ax1 = axs[0]
    ax2 = axs[1]
    ax1.plot(input_vs_roll[:,2], input_vs_roll[:,0], 'b', label='Throttle')
    ax1.plot(input_vs_roll[:,2], func_est(input_vs_roll[0,0],input_vs_roll[9,0], input_vs_roll[:,2], 2), 'g', label="Throttle Approximation")
    ax2.plot(input_vs_roll[:, 2], func_est(input_vs_roll[0, 1], input_vs_roll[9, 1], input_vs_roll[:, 2], 2), 'm', label="Pitch Approximation")
    ax2.plot(input_vs_roll[:,2], input_vs_roll[:,1], 'r', label='Pitch')

    ax1.legend(loc='upper right')
    plt.suptitle('Throttle and Pitch for max sink (0.0deg fpa and 26 m/s) and varying roll setpoint /w quadratic interpolation')
    ax1.set_ylabel('Throttle Command')
    ax1.set_xlabel('Roll Setpoint [rad]')

    ax2.set_ylabel('Pitch Command')
    ax2.legend(loc='upper left')
    plt.show()


    return L1


    # step=0
    # starting_input = [0.54, 0.0]
    # for i in range(1,6,1):  #everything except min sink and max sink
    #     res1 = Newton_Step(np.append(L1[0:2, i], [0.0]), starting_input, step)
    #     L1[2, i] = res1[0]
    #     L1[3, i] = res1[1]
    #
    #     res2 = Newton_Step(np.append(L2[0:2, i], [max_bank]), starting_input, step)
    #     L2[2, i] = res2[0]
    #     L2[3, i] = res2[1]
    #     step=0
    #

def get_L():
    return main()

if __name__ == '__main__':
    main()