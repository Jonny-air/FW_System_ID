import numpy as np


#Setup Newton
speed_tol = 0.001
fpa_tol = 0.0001
max_steps = 50

# Parameters
g =9.81
m = 3.204
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

def Newton_Step(desired_state, input, step):
    va, fpa, roll = desired_state
    u_t, pitch = input
    alpha = pitch - fpa
    n_p = n_slope*u_t + n_0
    # find current error
    if n_p == 0:  #catch division by zero
        T = 0
    else:
        T = rho * n_p**2 * D_p**4 * (CT[0] + CT[1] * va * np.cos(alpha - thrust_incl) / (n_p * D_p))
    L = 0.5 * rho * va**2 * (CL[0] + CL[1] * alpha)
    D = 0.5 * rho * va**2 * (CD[0] + CD[1] * alpha + CD[2] * alpha** 2)

    x1 = 1/m      *  (T * np.cos(alpha - thrust_incl) - D) - g * np.sin(fpa)
    x2 = 1/(m*va) * ((T * np.sin(alpha - thrust_incl) + L)*np.cos(roll) - m*g * np.cos(fpa))

    # return input if current error is low enough, otherwise continue
    if (abs(x1) < speed_tol and abs(x2) < fpa_tol):
        print(f"Converged after {step} steps")
        return np.array([[input[0]], [input[1]]])
    if step >= max_steps:
        print(f"Did not achieve desired accuracy after {step} steps. \n "
              f"The remaining error was {x1} for va_dot and {x2} for fpa_dot")
        return np.array([[input[0]], [input[1]]])

    # if it wasn't aborted we'll continue
    step += 1

    # find error gradient
    #thrust
    dTdU = rho*D_p**4*( 2*n_p*(CT[0] + CT[1]*(va*np.cos(alpha-thrust_incl))/(D_p*n_p))-(CT[1]*va*np.cos(alpha-thrust_incl)/n_p*D_p) ) * n_slope
    dx1dU = 1/m*np.cos(alpha-thrust_incl)*dTdU
    dx2dU = 1/(m*va)*np.sin(alpha-thrust_incl)*dTdU*np.cos(roll)

    #pitch
    dTda = -rho*n_p*D_p**3*CT[1]*va*np.sin(alpha-thrust_incl)
    dLda = 0.5*rho*va**2*CL[1]
    dDda = 0.5*rho*va**2*(CD[1]+2.0*CD[2]*alpha)
    dx1da = 1/m*(dTda*np.cos(alpha-thrust_incl)-T*np.sin(alpha-thrust_incl)-dDda)
    dx2da = np.cos(roll)/(m*va)*(dTda*np.sin(alpha-thrust_incl)+T*np.cos(alpha-thrust_incl)+dLda)

    # catch division by zero if gradients are zero - TODO later, for now assume these will probably never be zero zero

    # setup jakobian
    J = np.zeros((2,2))
    J[0,0]= dx1dU
    J[0,1]= dx1da
    J[1,0]= dx2dU
    J[1,1]= dx2da
    F = np.zeros((2,1))
    F[0] = -x1
    F[1] = -x2
    delta = np.linalg.solve(J, F)

    # calculate new inputs - Newton step
    u_t += delta[0, 0]     #mainly control fpa via airspeed
    pitch += delta[1, 0]   #mainly control airspeed via pitch
    # C1 = x1/dx1dU
    # C2 = x2/dx2dU
    # C3 = x1/dx1da
    # C4 = x2/dx2da
    # u_t -= 0.01*(x1/dx1dU + 0.1*x2/dx2dU)
    # pitch -= 0.1*(x1/dx1da + x2/dx2da)
    #constrain inputs
    u_t = max(min_thrust, min(max_thrust, u_t))
    pitch = max(min_pitch, min(max_pitch, pitch))
    input = [u_t, pitch]

    # call again
    return Newton_Step(desired_state, input, step)

if __name__ == '__main__':
    #setup tables
    L1 = np.zeros((4, 7))
    #       minsink    _p    max climb  crs    maxcr     maxsink   _p
    L1[0] = [  13.5,     13.5,    13.5,    14,     23,       23,      23]  #airspeed
    L1[1] = [ -0.13,   -0.095,  0.3791,   0.0,    0.0,   -0.174, -0.2094] #fpa
    L1[2] = [   0.0,        1,       1,     1,      1,        1,     0.0] # pitches for min sink and max sink

    L2 = L1

    step=0
    starting_input = [0.54, 0.0]
    for i in range(1,6,1):  #everything except min sink and max sink
        res1 = Newton_Step(np.append(L1[0:2, i], [0.0]), starting_input, step)
        L1[2, i] = res1[0]
        L1[3, i] = res1[1]

        res2 = Newton_Step(np.append(L2[0:2, i], [max_bank]), starting_input, step)
        L2[2, i] = res2[0]
        L2[3, i] = res2[1]
        step=0

    output_string = "{"
    for i in range(0, np.shape(L1)[0], 1):
        output_string += "\n{"
        for j in range(0, np.shape(L1)[1], 1):
            output_string += "%10.7ef, "%L1[i, j]
        output_string += "\b\b},"
    output_string += "\b\n},"
    for i in range(0, np.shape(L2)[0], 1):
        output_string += "\n{"
        for j in range(0, np.shape(L2)[1], 1):
            output_string += "%10.7ef, " % L2[i, j]
        output_string += "\b\b},"
    output_string += "\b\n}\n};"

    print(output_string)


#call Newton step as long as error is higher than x.
#desired state = v_a desired, fpa desired, roll