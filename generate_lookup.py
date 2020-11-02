import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import math


#Setup Newton
speed_tol = 0.001
fpa_tol = 0.0001
max_steps = 50
verbose = 0

# Parameters
g =9.81
m = 3.14
rho = 1.225
n_0 = 2.39 #rot per second
n_slope = 184.21 #rot per second input slope from 0-1
max_thrust = 1
min_thrust = 0
max_pitch = 30*np.pi/180
min_pitch = -30*np.pi/180
max_bank = 30*np.pi/180
thrust_incl = 0.0349 #rad
D_p = 0.28

CD = np.array(
    [0.0603, 0.34728666767791942, 0.2599156540313911]
)
CL = np.array(
    [0.52154, 2.8514436703100506]
)
CT = np.array(
    [0.1152, -0.1326]
)

def state_func(va, fpa, roll, alpha, u_t):
    n_p = n_slope * u_t + n_0
    # find current error
    if n_p == 0:  # catch division by zero
        T = 0
    else:
        T = rho * n_p ** 2 * D_p ** 4 * (CT[0] + CT[1] * va * np.cos(alpha - thrust_incl) / (n_p * D_p))
    L = 0.5 * rho * va ** 2 * (CL[0] + CL[1] * alpha)
    D = 0.5 * rho * va ** 2 * (CD[0] + CD[1] * alpha + CD[2] * alpha ** 2)

    x1 = 1 / m * (T * np.cos(alpha - thrust_incl) - D) - g * np.sin(fpa)
    x2 = 1 / (m * va) * ((T * np.sin(alpha - thrust_incl) + L) * np.cos(roll) - m * g * np.cos(fpa))
    return([x1, x2])

def setup_solver(x, case, input):
    if case == 0:
        alpha, u_t, roll = input
        va, fpa = x
        return state_func(va, fpa, roll, alpha, u_t)
    if case == 2:
        alpha, fpa, roll = input
        va, u_t = x
        return state_func(va, fpa, roll, alpha, u_t)
    if case == 6:
        va, u_t, roll = input
        fpa, pitch = x
        alpha = pitch - fpa
        return state_func(va, fpa, roll, alpha, u_t)
    elif case == 1 or case > 2:
        va, fpa, roll = input
        u_t, pitch = x
        alpha = pitch - fpa
        return state_func(va, fpa, roll, alpha, u_t)


def populate_table(L, roll):
    # min sink set alpha, throttle, solve for fpa and va -> case 1
    case = 0
    input = np.append(L[[0, 3], case], [roll])  # alpha, u_t, roll
    start_x = L[[1, 2], case]  # va, fpa
    roots = fsolve(setup_solver, start_x, args=(case, input))
    if np.isclose(setup_solver(roots, case, input), [0.0, 0.0]).all():
        va, fpa = roots[0:2]
        pitch = L[0,case] + fpa
        L[1,case] = va
        L[2,case] = fpa
        L[4,case] = pitch
    else:
        print(f"ERROR: Case {case} did not converge")

    case=1
    L[1, case] = L[1, 0]
    L[2, case] = L[2, 0] + (2*np.pi/180)
    input = np.append(L[[1, 2], case], [roll])  # va, fpa, roll
    start_x = L[[3, 4], case]  # u_t, pitch
    roots = fsolve(setup_solver, start_x, args=(case, input))
    if np.isclose(setup_solver(roots, case, input), [0.0, 0.0]).all():
        u_t, pitch = roots[0:2]
        L[3, case] = u_t
        L[4, case] = pitch
    else:
        print(f"ERROR: Case {case} did not converge")

    case = 2
    input = np.append(L[[0, 2], case], [roll])  # alpha, fpa, roll
    start_x = L[[1, 3], case]  # va, u_t
    roots = fsolve(setup_solver, start_x, args=(case, input))
    if np.isclose(setup_solver(roots, case, input), [0.0, 0.0]).all():
        va, u_t = roots[0:2]
        pitch = L[0, case] + L[2, case]
        L[4, case] = pitch
        L[1, case] = va
        L[3, case] = u_t
    else:
        print(f"ERROR: Case {case} did not converge")

    for case in range(3,6,1):
        input = np.append(L[[1, 2], case], [roll])  # va, fpa, roll
        start_x = L[[3, 4], case]  # u_t, pitch
        roots = fsolve(setup_solver, start_x, args=(case, input))
        if np.isclose(setup_solver(roots, case, input), [0.0, 0.0]).all():
            u_t, pitch = roots[0:2]
            L[3, case] = u_t
            L[4, case] = pitch
        else:
            print(f"ERROR: Case {case} did not converge")

    case = 6
    input = np.append(L[[1, 3], case], [roll])  # va, u_t, roll
    start_x = L[[2, 4], case]  # fpa, pitch
    roots = fsolve(setup_solver, start_x, args=(case, input))
    if np.isclose(setup_solver(roots, case, input), [0.0, 0.0]).all():
        fpa, pitch = roots[0:2]
        L[2, case] = fpa
        L[4, case] = pitch
    else:
        print(f"ERROR: Case {case} did not converge")


    return L


# def Newton_Step(desired_state, input, step):
#     va, fpa, roll = desired_state
#     u_t, pitch = input
#     alpha = pitch - fpa
#     n_p = n_slope*u_t + n_0
#     # find current error
#     if n_p == 0:  #catch division by zero
#         T = 0
#     else:
#         T = rho * n_p**2 * D_p**4 * (CT[0] + CT[1] * va * np.cos(alpha - thrust_incl) / (n_p * D_p))
#     L = 0.5 * rho * va**2 * (CL[0] + CL[1] * alpha)
#     D = 0.5 * rho * va**2 * (CD[0] + CD[1] * alpha + CD[2] * alpha** 2)
#
#     x1 = 1/m      *  (T * np.cos(alpha - thrust_incl) - D) - g * np.sin(fpa)
#     x2 = 1/(m*va) * ((T * np.sin(alpha - thrust_incl) + L)*np.cos(roll) - m*g * np.cos(fpa))
#
#     # return input if current error is low enough, otherwise continue
#     if (abs(x1) < speed_tol and abs(x2) < fpa_tol):
#         print(f"Converged after {step} steps")
#         return np.array([[input[0]], [input[1]]])
#     if step >= max_steps:
#         print(f"Did not achieve desired accuracy after {step} steps. \n "
#               f"The remaining error was {x1} for va_dot and {x2} for fpa_dot")
#         return np.array([[input[0]], [input[1]]])
#
#     # if it wasn't aborted we'll continue
#     step += 1
#
#     # find error gradient
#     #thrust
#     dTdU = rho*D_p**4*( 2*n_p*(CT[0] + CT[1]*(va*np.cos(alpha-thrust_incl))/(D_p*n_p))-(CT[1]*va*np.cos(alpha-thrust_incl)/n_p*D_p) ) * n_slope
#     dx1dU = 1/m*np.cos(alpha-thrust_incl)*dTdU
#     dx2dU = 1/(m*va)*np.sin(alpha-thrust_incl)*dTdU*np.cos(roll)
#
#     #pitch
#     dTda = -rho*n_p*D_p**3*CT[1]*va*np.sin(alpha-thrust_incl)
#     dLda = 0.5*rho*va**2*CL[1]
#     dDda = 0.5*rho*va**2*(CD[1]+2.0*CD[2]*alpha)
#     dx1da = 1/m*(dTda*np.cos(alpha-thrust_incl)-T*np.sin(alpha-thrust_incl)-dDda)
#     dx2da = np.cos(roll)/(m*va)*(dTda*np.sin(alpha-thrust_incl)+T*np.cos(alpha-thrust_incl)+dLda)
#
#     # catch division by zero if gradients are zero - TODO later, for now assume these will probably never be zero zero
#
#     # setup jakobian
#     J = np.zeros((2,2))
#     J[0,0]= dx1dU
#     J[0,1]= dx1da
#     J[1,0]= dx2dU
#     J[1,1]= dx2da
#     F = np.zeros((2,1))
#     F[0] = -x1
#     F[1] = -x2
#     delta = np.linalg.solve(J, F)
#
#     # calculate new inputs - Newton step
#     u_t += delta[0, 0]     #mainly control fpa via airspeed
#     pitch += delta[1, 0]   #mainly control airspeed via pitch
#     # C1 = x1/dx1dU
#     # C2 = x2/dx2dU
#     # C3 = x1/dx1da
#     # C4 = x2/dx2da
#     # u_t -= 0.01*(x1/dx1dU + 0.1*x2/dx2dU)
#     # pitch -= 0.1*(x1/dx1da + x2/dx2da)
#     #constrain inputs
#     u_t = max(min_thrust, min(max_thrust, u_t))
#     pitch = max(min_pitch, min(max_pitch, pitch))
#     input = [u_t, pitch]
#
#     # call again
#     return Newton_Step(desired_state, input, step)

if __name__ == '__main__':
    # set up tables
    #find alpha for Lift/Drag max
    a = -CL[1]*CD[2]
    b = -2*CL[0]*CD[2]
    c = CL[1]*CD[0] -CL[0]*CD[1]
    root = math.sqrt(b*b - 4*a*c)
    x1 = 1/(2.0*a)*((-b) + root) #generally we want to take the high alpha case for which we are slow, so x2 instead of this
    x2 = 1/(2.0*a)*((-b) - root)
    if verbose:
        print(f"max lift/drag at {x2*180/np.pi}")
        L = np.empty((0,2))
        D = np.empty((0, 2))
        for alpha in np.arange(-10*np.pi/180, 15*np.pi/180, 0.01):
            L = np.append(L, [[(CL[0] + CL[1] * alpha), alpha]], axis=0)
            D = np.append(D, [[(CD[0] + CD[1] * alpha + CD[2] * alpha**2), alpha]], axis=0)
        Y1 = (CL[0] + CL[1] * x1) / (CD[0] + CD[1] * x1 + CD[2] * x2 ** 2)
        Y2 = (CL[0] + CL[1] * x2) / (CD[0] + CD[1] * x2 + CD[2] * x2 ** 2)

        plt.figure(1, figsize=(10, 10))
        plt.plot(L[0:-1, 1] * 180 / np.pi, L[0:-1, 0], 'b', label='Lift')
        plt.plot(D[0:-1, 1] * 180 / np.pi, D[0:-1, 0], 'r', label='Drag')
        plt.plot(L[0:-1, 1] * 180 / np.pi, L[0:-1, 0]/D[0:-1, 0], 'g', label="Lift/Drag Ratio")
        plt.plot(x1*180/np.pi, Y1, '-o', Label="x1")
        plt.plot(x2*180/np.pi, Y2, '-o', Label="x2")

        plt.legend(loc='best')
        plt.title(f'Lift, Drag and Ratio of Coefficients')
        plt.grid()
        plt.show()

    L1 = np.zeros((5, 7))
    L2 = np.zeros((5,7))
    # table 0. correspond to not set values, everything else is needed or used as starting conditions
    #        minsink         _p    max climb      crs     maxcr      maxsink       _p
    L1[0] = [        x2,        x2,       x2,       0.,       0.,        0.,        0.]  # 0 alpha
    L1[1] = [  13.50000, 11.790514, 13.50000, 14.00000, 23.00000,  23.00000,  23.00000]  # 1 airspeed
    L1[2] = [ -0.130899, -0.095993, 0.349065, 0.000000, 0.000000, -0.174532, -0.209439]  # 2 fpa
    L1[3] = [   0.0,      0.353,    0.89,  0.55,   0.68,       0.,     0.0]  # 3 throttle for min sink and max sink
    L1[4] = [   0.0,      0.01,     0.0,   0.1,    0.1,     -0.1,     0.0]  # 4 pitch starting points

    L2[0] = [x2, x2, x2, 0., 0., 0., 0.]  # 0 alpha
    L2[1] = [13.50000, 11.790514, 13.50000, 14.00000, 23.00000, 23.00000, 23.00000]  # 1 airspeed
    L2[2] = [-0.130899, -0.095993, 0.349065, 0.000000, 0.000000, -0.174532, -0.209439]  # 2 fpa
    L2[3] = [0.0, 0.353, 0.89, 0.55, 0.68, 0., 0.0]  # 3 throttle for min sink and max sink
    L2[4] = [0.0, 0.01, 0.0, 0.1, 0.1, -0.1, 0.0]  # 4 pitch starting points

    roll = 0.0
    L1 = populate_table(L1, roll)

    roll = max_bank
    L2 = populate_table(L2, roll)

    output_string = "{"
    for i in range(1, np.shape(L1)[0], 1):
        output_string += "\n{"
        for j in range(0, np.shape(L1)[1], 1):
            output_string += "%10.7ef, " % L1[i, j]
        output_string += "\b\b},"
    output_string += "\b\n},"
    for i in range(1, np.shape(L2)[0], 1):
        output_string += "\n{"
        for j in range(0, np.shape(L2)[1], 1):
            output_string += "%10.7ef, " % L2[i, j]
        output_string += "\b\b},"
    output_string += "\b\n}\n};"

    print(output_string)



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
