#insert ode's
#with given pitch and throttle, solve ode until steady state is reached
#iterate over pitches and throttles
#generate lookup table
import numpy as np
import matplotlib.pyplot as plt

#Simulation Parameters
single_sim = 1

#if single sim these will be used:
pitch_s = 3 *np.pi/180
thrust_s = 0.0

#otherwise these will be used
pitch_start = 0.2
pitch_end = 0.0
pitch_step = 0.0
thrust_start = 0.0
thrust_end = 0.0
thrust_step = 0.01

tstep = 0.02            # Sampling time (sec)
simulation_time = 30    # Length of time to run simulation (sec)

#Variables
g =9.81
m = 3.7
rho = 1.25
n_0 = 2.39 #rot per second
n_slope = 184.21 #rot per second input slope from 0-1
thrust_incl = 0.0 #rad
D_p = 0.28
CD = np.array([-0.023616004820951813, -0.0707196273581569, -2.353334838897727])
CL = np.array([0.48944183370577127, 14.815092813394312])
CT = np.array([0.115, -0.1326])


# 4th Order Runge Kutta Calculation
def RK4(x, dt, input):
    # Inputs: x[k], u[k], dt (time step, seconds)
    # Returns: x[k+1]

    # Calculate slope estimates
    K1 = stateDerivative(x, input)
    K2 = stateDerivative(x + K1 * dt / 2, input)
    K3 = stateDerivative(x + K2 * dt / 2, input)
    K4 = stateDerivative(x + K3 * dt, input)

    # Calculate x[k+1] estimate using combination of slope estimates
    x_next = x + 1 / 6 * (K1 + 2 * K2 + 2 * K3 + K4) * dt

    return x_next

# Simulation time and model parameters

def stateDerivative(x, input):

    pitch, u_t = input
    v_a =x[0]
    fpa =x[1]
    print(fpa)

    xdot=np.zeros(2)

    #Lift, Drag Thrust
    alpha = pitch - fpa
    n_p = n_slope*u_t+n_0
    L = 0.5*rho*v_a**2*(CL[0] + CL[1]*alpha)
    D = 0.5*rho*v_a**2*(CD[0] + CD[1]*alpha + CD[2]*alpha**2)
    T = rho*n_p**2*D_p**4*(CT[0] + CT[1]*v_a*np.cos(alpha - thrust_incl)/(n_p*D_p))


    xdot[0] = 1/m*(T*np.cos(alpha-thrust_incl)-D)-g*np.sin(fpa)

    if v_a <= 0.0:
        xdot[1] = 0.0
    else:
        xdot[1] = 1/(m*v_a)*((T*np.sin(alpha-thrust_incl)+L)-m*g*np.cos(fpa)) #include ROLL later

    return xdot

def get_ss(pitch, u_t):
    input = [pitch, u_t]
    t = np.arange(0, simulation_time, tstep)  # time array
    x = np.zeros((2, np.size(t)))
    x[0,0] = 14
    x[1,0] = pitch

    for k in range(0, np.size(t) - 1):
        # Predict state after one time step
        x[1, k] % (2 * np.pi)
        print(x[1,k])
        x[:, k + 1] = RK4(x[:, k], tstep, input)

    ss_va = x[0, np.size(t) - 1]
    ss_fpa = x[1, np.size(t) - 1]
    rt_va, rt_fpa = 0.0, 0.0

    # rise time and steady state speed
    for k in range(0, np.size(t) - 1):
        if abs(x[0, k] - ss_va) < abs(0.1 * ss_va):
            rt_va = k * tstep
            break

    for k in range(0, np.size(t) - 1):
        if abs(x[1, k] - ss_fpa) < abs(0.1 * ss_fpa):
            rt_fpa= k * tstep
            break

    if single_sim:
        plt.figure(1, figsize=(10, 10))
        plt.plot(t[0:-1], x[0, 0:-1], 'b', label='v_a [m/s]')
        plt.plot(t[0:-1], x[1, 0:-1] * 180 / np.pi, 'r', label='fpa [DEG]')

        plt.xlabel('Time (sec)')
        plt.ylabel('Velocity')
        plt.legend(loc='best')
        plt.title(f'Velocity with pitch: {pitch*180/np.pi} deg and u_t: {u_t}')
        plt.figtext(.6, .75, f"Steady state v_a = {round(ss_va,2)} m/s in {round(rt_va,1)} sec \n"
                            f"Steady state fpa = {round(ss_fpa*180/np.pi,2)} deg in {round(rt_fpa,1)} sec")
        plt.grid()
        plt.show()

    return np.array([ss_va, ss_fpa, rt_va, rt_fpa, pitch, u_t])

#n=8000/60 #RPS
#Dp=0.0254*14
#A_drone=0.25 #m^2
#m = 3.2


    '''plt.figure(1, figsize=(10, 10))
    plt.plot(t[0:-1], x[0, 0:-1]*3.6, 'b', label='v [km/h]')
    plt.plot(t[0:-1], x[1, 0:-1] * 180 / np.pi, 'r', label='theta [DEG]')
    plt.plot(t[0:-1], x[2, 0:-1]*3.6, 'g', label='w [km/h]')

    plt.xlabel('Time (sec)')
    plt.ylabel('Velocity')
    plt.legend(loc='best')
    plt.title('Velocity with theta = '+str(int(theta* 180 / np.pi))+'DEG')

    plt.show()'''

#print(get_ssv(30*np.pi/180, m, A_drone, Dp, n))

'''
plt.figure(1, figsize=(10,10))
plt.plot(t[0:-1],x[0,0:-1],'b',label='v')
plt.plot(t[0:-1],x[1,0:-1]*180/np.pi,'r',label='theta')
plt.plot(t[0:-1],x[2,0:-1],'g',label='w')

plt.xlabel('Time (sec)')
plt.ylabel('Velocity')
plt.legend(loc='best')
plt.title('Time History of Control Inputs')

plt.show()
'''

if __name__ == '__main__':
    if single_sim:
        get_ss(pitch_s, thrust_s)
    else:
        pitches = np.arange(pitch_start, pitch_end+pitch_step, pitch_step)
        thrusts = np.arange(thrust_start, thrust_end+thrust_step, thrust_step)
        steady_states = np.empty((0,6))
        for p in pitches:
            for t in thrusts:
                steady_states = np.append(steady_states, [get_ss(p, t)], axis=0)