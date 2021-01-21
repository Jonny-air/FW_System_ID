import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.optimize import fmin
import math
import Calc_Coefficients
from load_model import get_params
from Lift_Calculation import get_lift
from Drag_Calculation import get_drag
from visualize_lookup import visualize
import Thrust_Calculation

#Setup Coefficient Calculation
_model_name = "Believer_201112_true"
_verbose = False
_show_plots = False
_fix_cd2 = True #adjusted manually to get the minimum drag at a realistic alpha

# Environment Parameters
g =9.80
rho = 1.225
_env = g, rho

#other data for lookup not contained in model
max_speed = 26
min_speed = 9
cruise_speed = 16
max_fpa = 20*np.pi/180 #20deg
max_rps = 1000/(2*np.pi)
max_thrust = 1
min_thrust = 0
max_pitch = 30*np.pi/180
min_pitch = -30*np.pi/180
max_bank = 45*np.pi/180
roll_elev_gain = 0.2

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

CL, CD = Calc_Coefficients.calc_lift_drag(_model_name, _fix_cd2, verbose=_verbose, show_plots=_show_plots, env=_env)
CT = Calc_Coefficients.calc_thrust(_model_name, CL, CD, verbose=_verbose, show_plots=_show_plots, env=_env)
CE = Calc_Coefficients.calc_elev(_model_name, verbose=_verbose, show_plots=_show_plots)
[m, n_0, n_slope, thrust_incl, D_p, actuator_control_ch, elev_control_ch, state_level] = get_params(_model_name, verbose=_verbose) #defined for the model via Add model

def state_func(va, fpa, roll, alpha, u_t):
    n_p = n_slope * u_t + n_0
    # find current error
    if(n_p == 0):
        T = 0
    else:
        T = Thrust_Calculation.get_thrust(CT, va, alpha, n_p, thrust_incl, D_p, rho=rho)

    L = get_lift(CL,va,alpha, rho=rho)
    D = get_drag(CD,va,alpha, rho=rho)

    x1 = 1 / m * (T * np.cos(alpha - thrust_incl) - D) - g * np.sin(fpa)
    x2 = 1 / (m * va) * ((T * np.sin(alpha - thrust_incl) + L) * np.cos(roll) - m * g * np.cos(fpa))
    return([x1, x2])

def setup_solver(x, case, input):
    if case == 0 or case == 1:
        alpha, u_t, roll = input
        va, fpa = x
        return state_func(va, fpa, roll, alpha, u_t)
    if case == 2:
        alpha, fpa, roll = input
        va, u_t = x
        return state_func(va, fpa, roll, alpha, u_t)
    if case == 3 or case == 4:
        va, fpa, roll = input
        u_t, pitch = x
        alpha = pitch - fpa
        return state_func(va, fpa, roll, alpha, u_t)
    if case == 5 or case == 6:
        u_t, va, roll = input
        fpa, pitch = x
        alpha = pitch - fpa
        return state_func(va, fpa, roll, alpha, u_t)

def populate_table(L, roll, maxcl3cd2, ldmax):
    # min sink set alpha to maxcl3cd2, throttle=0, solve for fpa and va -> case 0
    case = 0
    alpha = maxcl3cd2
    u_t = 0.0
    input = [alpha, u_t, roll]  # alpha, u_t, roll
    start_x = L[[1, 2], case]  # va, fpa
    roots = least_squares(setup_solver, start_x, args=(case, input), bounds=((min_speed, -max_fpa),(max_speed,max_fpa))).x

    if np.isclose(setup_solver(roots, case, input), [0.0, 0.0]).all():
        va, fpa = roots[0:2]
        pitch = alpha + fpa
        L[1,case] = va
        L[2,case] = fpa
        L[3,case] = u_t
        L[4,case] = pitch
    else:
        print(f"[ERROR] Case {case} with roll {roll} did not converge")

    case = 1
    # min sink + p should be just like case 0 but with nonzero throttle
    alpha = maxcl3cd2
    n_p = Thrust_Calculation.get_highest_np(CT, L[1, 0], alpha, thrust_incl, D_p, n_0)
    u_t = (n_p - n_0)/n_slope
    input = [alpha, u_t, roll]  # alpha, u_t, roll
    start_x = L[[1, 2], case]  # va, fpa
    roots = least_squares(setup_solver, start_x, args=(case, input), bounds=((min_speed, -max_fpa),(max_speed, max_fpa))).x
    if np.isclose(setup_solver(roots, case, input), [0.0, 0.0]).all():
        va, fpa = roots[0:2]
        pitch = alpha + fpa
        L[1, case] = va
        L[2, case] = fpa
        L[3, case] = u_t
        L[4, case] = pitch
    else:
        print(f"[ERROR] Case {case} with roll {roll} did not converge")

    #max climb - set max fpa and alpha = ldmax
    case = 2
    alpha = ldmax
    fpa = max_fpa
    input = [alpha, fpa, roll]  # alpha, fpa, roll
    start_x = L[[1, 3], case]  # va, u_t
    roots = least_squares(setup_solver, start_x, args=(case, input), bounds=((min_speed, 0.0),(max_speed,1.0))).x
    if np.isclose(setup_solver(roots, case, input), [0.0, 0.0]).all():
        va, u_t = roots[0:2]
        pitch = alpha + fpa
        L[1, case] = va
        L[2, case] = fpa
        L[3, case] = u_t
        L[4, case] = pitch
    else:
        print(f"[ERROR] Case {case} with roll {roll} did not converge")

    #cruise
    case = 3
    va = cruise_speed
    fpa = 0.0
    input = [va, fpa, roll]  # va, fpa, roll
    start_x = L[[3, 4], case]  # u_t, pitch
    roots = least_squares(setup_solver, start_x, args=(case, input), bounds=((0.0, min_pitch), (1.0, max_pitch))).x
    if np.isclose(setup_solver(roots, case, input), [0.0, 0.0]).all():
        u_t, pitch = roots[0:2]
        L[1, case] = va
        L[2, case] = fpa
        L[3, case] = u_t
        L[4, case] = pitch
    else:
        print(f"[ERROR] Case {case} with roll {roll} did not converge")

    #max cruise
    case = 4
    va = max_speed
    fpa = 0.0
    input = [va, fpa, roll]  # va, fpa, roll
    start_x = L[[3, 4], case]  # u_t, pitch
    roots = least_squares(setup_solver, start_x, args=(case, input), bounds=((0.0, min_pitch), (1.0, max_pitch))).x
    if np.isclose(setup_solver(roots, case, input), [0.0, 0.0]).all():
        u_t, pitch = roots[0:2]
        L[1, case] = va
        L[2, case] = fpa
        L[3, case] = u_t
        L[4, case] = pitch
    else:
        print(f"[ERROR] Case {case} with roll {roll} did not converge")

    #max sink
    case = 6
    u_t = 0.0
    va = max_speed
    input = [u_t, va,  roll] # u_t, va, roll
    start_x = L[[2, 4], case]  # fpa, pitch
    roots = least_squares(setup_solver, start_x, args=(case, input), bounds=((-1, -1),(1, 1))).x #bounds are less conservative since solver might overshoot and come back within bounds
    if np.isclose(setup_solver(roots, case, input), [0.0, 0.0]).all():
        fpa, pitch = roots[0:2]
        L[1, case] = va
        L[2, case] = fpa
        L[3, case] = u_t
        L[4, case] = pitch
    else:
        print(f"[ERROR] Case {case} with roll {roll} did not converge")

    #max sink + p
    case = 5
    alpha = L[4, 6] - L[2, 5] #pitch minus fpa
    va = max_speed
    n_p = Thrust_Calculation.get_highest_np(CT, va, alpha, thrust_incl, D_p, n_0)
    u_t = (n_p - n_0) / n_slope
    input = [u_t, va,  roll] # u_t, va, roll
    start_x = L[[2, 4], case]  # fpa, pitch
    roots = least_squares(setup_solver, start_x, args=(case, input), bounds=((-1, -1),(1, 1))).x
    if np.isclose(setup_solver(roots, case, input), [0.0, 0.0]).all():
        fpa, pitch = roots[0:2]
        L[1, case] = va
        L[2, case] = fpa
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
    if _verbose:
        print('Finding maxcl3cd2...')
    maxcl3cd2 = fmin(cl3cd2, 0.0)[0]
    if _verbose:
        print(f"max lift/drag at {ldmax*180/np.pi}")
        print(f"max cl3cd2 at {maxcl3cd2*180/np.pi}")
        L = np.empty((0,2))
        D = np.empty((0, 2))
        for alpha in np.arange(-5*np.pi/180, 15*np.pi/180, 0.01):
            L = np.append(L, [[(CL[0] + CL[1] * alpha), alpha]], axis=0)
            D = np.append(D, [[(CD[0] + CD[1] * alpha + CD[2] * alpha**2), alpha]], axis=0)
        Y1 = (CL[0] + CL[1] * x1) / (CD[0] + CD[1] * x1 + CD[2] * ldmax ** 2)
        Y2 = (CL[0] + CL[1] * ldmax) / (CD[0] + CD[1] * ldmax + CD[2] * ldmax ** 2)
        Y3 = (CL[0] + CL[1] * maxcl3cd2) / (CD[0] + CD[1] * maxcl3cd2 + CD[2] * maxcl3cd2 ** 2)

        plt.figure(1, figsize=(7, 7))
        plt.plot(L[0:-1, 1] * 180 / np.pi, L[0:-1, 0]/D[0:-1, 0], 'g', label="Lift/Drag Ratio")
        #plt.plot(x1*180/np.pi, Y1, '-o', Label="x1")
        plt.plot(maxcl3cd2*180/np.pi, Y2, '-o', Label=r'Max $L^3 D^{-2}$')
        plt.plot(ldmax*180/np.pi, Y2, '-o', Label=r'Max $L D^{-1}$')
        plt.ylabel("Ratio of Lift to Drag", color = 'green')
        plt.xlabel(r"Angle of attack: $\alpha, [\mathrm{deg}]$")

        plt.grid()
        plt.legend(loc='upper left')
        ax = plt.twinx()
        ax.plot(L[0:-1, 1] * 180 / np.pi, L[0:-1, 0], 'b', label='Lift Coefficient')
        ax.plot(D[0:-1, 1] * 180 / np.pi, D[0:-1, 0], 'r', label='Drag Coefficient')
        ax.set_ylabel(r"Lift and Drag Coeffiecients, $[m^2]$", color='black')
        ax.legend(loc='upper right')
        plt.title(f'Lift, Drag and Ratio of Coefficients')
        plt.show()

    L1 = np.zeros((5, 7))
    L2 = np.zeros((5,7))
    # table 0. correspond to not set values, everything else is needed or used as starting conditions (check mxcl3cd2 for reasonable value and set maxcr high enough)

    #Believer   thr, alpha <-- +1degfpa --      alpha, fpa - alpha, fpa --   as, fpa --     as, alpha       as, thr
    # L1[0] = [  maxcl3cd2,     ldmax,          ldmax,      0.034906,       0.002,          ldmax,          0.          ]  # 0 alpha
    # L1[1] = [  9.50000,       9.790514,       13.50000,   17.00000,       26.00000,       26.00000,       26.00000    ]  # 1 airspeed
    # L1[2] = [ -0.130899,      -0.095993,      0.349065,   0.000000,       0.000000,       -0.174532,      -0.209439   ]  # 2 fpa
    # L1[3] = [   0.0,          0.5,            0.89,       0.55,           0.68,           0.5,            0.0         ]  # 3 throttle for min sink and max sink
    # L1[4] = [   0.0,          0.01,           0.5,        0.1,            0.1,            -0.2,           -0.2        ]  # 4 pitch starting points
    #
    #
    # L2[0] = [  maxcl3cd2,     ldmax,       ldmax,    0.034906,    0.002,     ldmax,         0.]  # 0 alpha
    # L2[1] = [  9.5, 9.790514, 13.50000, 11.00000, 15.00000,  15.00000,   15.00000]  # 1 airspeed
    # L2[2] = [ -0.130899, -0.095993, 0.027925, 0.000000, 0.000000, -0.174532,  -0.209439]  # 2 fpa
    # L2[3] = [   0.0,       0.353,     0.89,      0.55,    0.68,       0.3,        0.0]  # 3 throttle for min sink and max sink
    # L2[4] = [   0.0,        0.01,     0.5,        0.1,     0.1,       -0.2,      -0.2]  # 4 pitch starting points


    # Easyglider
    #        minsink         _p    max climb      crs     maxcr      maxsink_p     maxsink
    L1[0] = [  maxcl3cd2,     ldmax,       ldmax,    0.034906,    0.002,     ldmax,         0.]  # 0 alpha
    L1[1] = [  9.50000, 9.790514, 13.50000, 11.00000, 17.00000,  17.00000,   17.00000]  # 1 airspeed
    L1[2] = [ -0.130899, -0.095993, 0.349065, 0.000000, 0.000000, -0.174532,  -0.209439]  # 2 fpa
    L1[3] = [   0.0,       0.353,     0.89,      0.55,    0.68,       0.3,        0.0]  # 3 throttle for min sink and max sink
    L1[4] = [   0.0,        0.01,     0.5,        0.1,     0.1,       -0.2,      -0.2]  # 4 pitch starting points


    L2[0] = [  maxcl3cd2,     ldmax,       ldmax,    0.034906,    0.002,     ldmax,         0.]  # 0 alpha
    L2[1] = [  9.5, 9.790514, 13.50000, 11.00000, 15.00000,  15.00000,   15.00000]  # 1 airspeed
    L2[2] = [ -0.130899, -0.095993, 0.027925, 0.000000, 0.000000, -0.174532,  -0.209439]  # 2 fpa
    L2[3] = [   0.0,       0.353,     0.89,      0.55,    0.68,       0.3,        0.0]  # 3 throttle for min sink and max sink
    L2[4] = [   0.0,        0.01,     0.5,        0.1,     0.1,       -0.2,      -0.2]  # 4 pitch starting points

    roll = 0.0
    L1 = populate_table(L1, roll, maxcl3cd2, ldmax)

    roll = max_bank
    L2 = populate_table(L2, roll, maxcl3cd2, ldmax)

    #add elevator lookup
    vas = L1[1,:]
    pitches = L1[4,:]
    elevs = CE[0]*pitches*(CE[1]*vas+CE[2]*vas**2)
    L1 = np.append(L1, [elevs], axis=0)

    vas = L2[1, :]
    pitches = L2[4, :]
    elevs = CE[0] * pitches * (CE[1] * vas + CE[2] * vas ** 2) + max_bank*roll_elev_gain
    L2 = np.append(L2, [elevs], axis=0)




    output_string = "{"
    for i in range(1, np.shape(L1)[0]-1, 1):
        output_string += "\n{"
        for j in range(0, np.shape(L1)[1], 1):
            output_string += "%10.7ef, " % L1[i, j]
        output_string += "\b\b},"
    output_string += "\b\n},"
    output_string += "\n{ /*roll = max */"
    for i in range(1, np.shape(L2)[0]-1, 1):
        output_string += "\n{"
        for j in range(0, np.shape(L2)[1], 1):
            output_string += "%10.7ef, " % L2[i, j]
        output_string += "\b\b},"
    output_string += "\b\n}\n};"

    print(output_string)


    os2  = "    %10.7ef, // 0 lift coefficient at zero angle of attack \n" % CL[0]
    os2 += "    %10.7ef, // 1 lift curve slope \n" % CL[1]
    os2 += "    %10.7ef, // 2 drag coefficient at zero angle of attack \n" % CD[0]
    os2 += "    %10.7ef, // 3 drag coefficient w.r.t. angle of attack \n" % CD[1]
    os2 += "    %10.7ef, // 4 drag coefficient w.r.t. angle of attack squared \n" % CD[2]
    os2 += "    %10.7ef, // 5 throttle zero inflow coefficient \n" % CT[0]
    os2 += "    %10.7ef, // 6 throttle advance ratio coefficient \n" % CT[1]
    os2 += "    %10.7ef, // 7 rps at zero throttle input \n" % (n_0)
    os2 += "    %10.7ef, // 8 rps to throttle input slope \n" % (n_slope)
    os2 += "    %10.7ef, // 9 aircraft mass \n" % m
    os2 += "    %10.7ef, // 10 aircraft propeller diameter\n" % D_p
    os2 += "    %10.7ef, // 11 aircraft thrust inclination angle \n" % thrust_incl
    os2 += "    %10.7ef // 12 max rps \n" % (max_rps)
    os2 += "};"

    print(os2)

    print('\n\n\n\n')

    output_string = "{"
    for i in [1, 4, 5]:
        output_string += "\n{"
        for j in range(0, np.shape(L1)[1], 1):
            output_string += "%10.7ef, " % L1[i, j]
        output_string += "\b\b},"
    output_string += "\b\n},"
    output_string += "\n{ /*roll = max */"
    for i in [1, 4, 5]:
        output_string += "\n{"
        for j in range(0, np.shape(L2)[1], 1):
            output_string += "%10.7ef, " % L2[i, j]
        output_string += "\b\b},"
    output_string += "\b\n}\n};"

    print(output_string)

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
    L1 = main()
    visualize(L1, _model_name)
