from Import_Datapoints import import_data
from scipy.optimize import least_squares
from Drag_Calculation import get_drag
from Lift_Calculation import get_lift
from Thrust_Calculation import get_thrust
from load_model import get_params
from Coefficient_plots import plot_lift_drag
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as cols
import matplotlib.cm as cm
import pickle

#Setup
_model_name = 'Believer_201112_true'
_fix_cd2 = False
_verbose = True
_show_plots = True
_env = [9.8, 1.225]
_cd2 = 6.8664156540313911

def calc_elev(model_name, verbose = False, show_plots = False, show_titles=False):
    # Lift Drag
    try:
        if verbose:
            print("Loading ID data for the elevator coefficients...")
        vas, va_dots, fpas, fpa_dots, alphas, rolls, pitches, elevs = import_data(model_name, 0, ramp_types=[0,2], verbose=verbose) #manual ramps or pitch ramps are okay, elevator ramps have too high pitchrate
    except:
        print("[ERROR] No Data for the elevator ID found")
        return 0

    def Elev_func(CE):
        E = CE[0]*pitches*(CE[1]*vas+CE[2]*vas**2)
        err = E - elevs
        # weigh alphas around 0 higher
        # err *= np.cos(15 * alphas)
        return err.transpose()[0]

    start_CE = np.array([0.1, 0.1, 0.1])
    sol = least_squares(Elev_func, start_CE)
    CE = sol.x
    resE= sol.cost

    if show_plots:
        pred_elevs = CE[0]*pitches*(CE[1]*vas+CE[2]*vas**2)

        t = range(0, elevs.shape[0], 1)
        fig, axs = plt.subplots(2, figsize=(7, 7))

        axs[0].plot(t, elevs, 'b', label='Commanded elevator')
        axs[1].plot(t, pitches*180/np.pi, 'b', label='Measured pitch')
        axs[0].plot(t, pred_elevs, 'g', label='Predicted elevator')

        axs[0].set_xlabel('Datapoints')
        axs[1].set_xlabel('Datapoints')
        axs[0].set_ylabel('Elevator command [%]')
        axs[1].set_ylabel('Pitch [deg]')
        if show_titles: axs[0].set_title(f'Comparison predicted vs actual elevator commands')
        axs[0].grid()
        axs[1].grid()
        axs[0].legend(loc='best')
        axs[1].legend(loc='best')
        plt.tight_layout()
        plt.show()

    return CE

def calc_lift_drag(model_name, fix_cd2, verbose = False, show_plots = False, show_titles = False, env = _env, cd2 = _cd2):
    # get parameters
    m = get_params(model_name)[0]
    g, rho = env

    #Lift Drag
    try:
        if verbose:
            print("Loading ID data for the lift/drag coefficients...")
        vas, va_dots, fpas, fpa_dots, alphas, rolls, pitches, elevs = import_data(model_name, 0, verbose=verbose) #you can pass des ramp types as array ramp_types= [0,1,2]
    except:
        print("[ERROR] No Data for the Lift/Drag Maneuver found")
        return 0

    def Lift_func(CL):
        L = 0.5 * rho * vas ** 2 * (CL[0] + CL[1] * alphas)
        err = -fpa_dots + 1/m*(L*np.cos(rolls)-m*g*np.cos(fpas)) #fpa_dots: fpa_dot*va for shortness
        #weigh alphas around 0 higher XXX do we need to correct for scaling with airspeed here?
        # err *= np.cos(15*alphas)
        return err.transpose()[0]

    start_CL = np.array([0.10723261045844895, 1.9946490526929737])
    sol = least_squares(Lift_func, start_CL, bounds=((0, 0), (1, 5)))
    CL = sol.x
    resL = sol.cost

    # setup least squares for Drag
    if fix_cd2:
        def Drag_func(CD, CD2):
            CD = np.append(CD, CD2)
            # D = 0.5 * rho * vas ** 2 * (CD[0] + CD[1] * alphas + CD2 * alphas ** 2)
            D = get_drag(CD, vas, alphas)
            err = -va_dots + 1 / m * (-D) - g * np.sin(fpas)
            return err.transpose()[0]

        start_CD = np.array([0.02697635475641432, 0.2328666767791942])
        CD2 = (cd2,)
        sol = least_squares(Drag_func, start_CD, args=(CD2), bounds=((0, 0), (1, 1)))
        CD = sol.x
        CD = np.append(CD, CD2)
        resD = sol.cost
    else:
        def Drag_func(CD):
            D = get_drag(CD, vas, alphas)
            err = -va_dots + 1 / m * (-D) - g * np.sin(fpas)
            return err.transpose()[0]

        start_CD = np.array([0.02697635475641432, 0.2328666767791942, 0.68664156540313911])
        sol = least_squares(Drag_func, start_CD, bounds=((0, 0, 0), (1, 1, 5)))
        CD = sol.x
        resD = sol.cost



    if show_plots:
        L = get_lift(CL, vas, alphas)
        D = get_drag(CD, vas, alphas)
        pred_va_dot = 1 / m * (-D) - g * np.sin(fpas)
        pred_fpa_dot = 1 / m * (L * np.cos(rolls) - m * g * np.cos(fpas))

        t = range(0, fpas.shape[0], 1)
        fig, axs = plt.subplots(2, figsize=(7, 7))

        axs[0].plot(t, va_dots, 'b', label=r'Measured $\dot{v}_A$')
        axs[1].plot(t, fpa_dots, 'b', label=r'Measured $\dot{\gamma} \cdot v_A$')
        axs[0].plot(t, pred_va_dot, 'g', label=r'Predicted $\dot{v}_A$')
        axs[1].plot(t, pred_fpa_dot, 'g', label=r'Predicted $\dot{\gamma} \cdot v_A$')

        axs[0].set_xlabel('Datapoints')
        axs[1].set_xlabel('Datapoints')
        axs[0].set_ylabel(r'Acceleration: $\dot{v}_A$ $[\frac{m}{s^2}]$')
        axs[1].set_ylabel(r'Acceleration: $\dot{\gamma} \cdot v_A$ $[\frac{m}{s^2}]$')
        if show_titles: axs[0].set_title(f'Comparison predicted vs actual derivatives')
        axs[0].grid()
        axs[1].grid()
        axs[0].legend(loc='best')
        axs[1].legend(loc='best')
        plt.tight_layout()
        plt.show()


        #Lift Drag Coefficients
        # datapoints
        lift_datapoints = 2 * m / (rho * vas ** 2) * (fpa_dots + g * np.cos(fpas)) #fpa_dots: fpa_dot*va for shortness
        drag_datapoints = -2 * m / (rho * (vas ** 2)) * (va_dots + g * np.sin(fpas))
        datapoints = [alphas, lift_datapoints, drag_datapoints]
        plot_lift_drag(model_name, CL, CD, datapoints, resL, resD, fix_cd2, show_titles=show_titles)
    return CL, CD

def calc_thrust(model_name, CL, CD, verbose = False, show_plots = False, show_titles = False, env = [9.8, 1.225]):
    # get parameters
    [m, n_0, n_slope, thrust_incl, D_p] = get_params(model_name)[0:5]
    g, rho = env
    # Thrust
    try:
        if verbose:
            print("Loading ID data for the thrust coefficient...")
        vas, va_dots, fpas, fpa_dots, alphas, rolls, pitches, elevs, n_ps = import_data(model_name, 1, verbose=verbose)
    except:
        print("[ERROR] No Data for the Thrust Maneuver found")
        return 0

    indices = np.nonzero(n_ps > 30)[0]
    # setup least squares
    def Thrust_func(CT):
        T = get_thrust(CT, vas, alphas, n_ps, thrust_incl, D_p, rho=rho)
        D = get_drag(CD, vas, alphas, rho=rho)
        err = -va_dots + 1 / m * (T * np.cos(alphas - thrust_incl) - D) - g * np.sin(fpas)
        #L = get_lift(CL, vas, alphas, rho=rho)
        #err2 = -fpa_dots + 1 / m * (T * np.sin(alphas - thrust_incl) + L) * np.cos(rolls) - g * np.cos(fpas)
        # err = np.sqrt(err1**2 + err2**2)
        err = err[indices]
        # weigh datapoints around normal operation point heavier
        #err *= np.cos((n_ps - 140) * 0.0125)  # times approx 0.5 at the lower n_ps
        return err.transpose()[0]

    start_CT = np.array([0.20245296276934735, -0.2993443648040376])
    sol = least_squares(Thrust_func, start_CT, loss="cauchy", f_scale=1.0, bounds=((-3, -3), (3, 3))) #TODO make this variable
    CT = sol.x
    resT = sol.cost

    if show_plots:
        CT = [0.23079, -0.2823]
        D = get_drag(CD, vas, alphas, rho=rho)
        L = get_lift(CL, vas, alphas, rho=rho)
        T = get_thrust(CT, vas, alphas, n_ps, thrust_incl, D_p, rho=rho)
        t0 = rho * n_ps ** 2 * D_p ** 4
        denom1 = np.cos(alphas - thrust_incl) * t0
        thrust_datapoints = np.divide(m * (va_dots + g * np.sin(fpas)) + D, denom1, out=np.zeros_like(va_dots), where=n_ps!=0)
        pred_va_dot = 1 / m * (T * np.cos(alphas - thrust_incl) - D) - g * np.sin(fpas)
        pred_fpa_dot = 1 / m * (T * np.sin(alphas - thrust_incl) + L) * np.cos(rolls) - g * np.cos(fpas)
        t = range(0, fpas.shape[0], 1)

        fig, axs = plt.subplots(2, figsize=(7,7))
        axs[0].plot(t, va_dots, 'b', label=r'Measured $\dot{v}_A$')
        axs[1].plot(t, fpa_dots, 'b', label=r'Measured $\dot{\gamma} \cdot v_A$')
        axs[0].plot(t, pred_va_dot, 'g', label=r'Predicted $\dot{v}_A$')
        axs[1].plot(t, pred_fpa_dot, 'g', label=r'Predicted $\dot{\gamma} \cdot v_A$')

        axs[0].set_xlabel('Datapoints')
        axs[1].set_xlabel('Datapoints')
        axs[0].set_ylabel(r'Acceleration: $\dot{v}_A$ $[\frac{m}{s^2}]$')
        axs[1].set_ylabel(r'Acceleration: $\dot{\gamma} \cdot v_A$ $[\frac{m}{s^2}]$')
        if show_titles: axs[0].set_title(f'Comparison predicted vs actual accelerations')
        axs[0].grid()
        axs[1].grid()
        axs[0].legend(loc='best')
        axs[1].legend(loc='best')
        plt.tight_layout()
        plt.show()

        # Thrust Coefficient Plot
        indices = np.nonzero(n_ps > 30)[0]
        thrust_datapoints = thrust_datapoints[indices]
        data_inflows = vas * np.cos(alphas - thrust_incl)
        data_inflows = data_inflows[indices]
        n_ps = n_ps[indices]

        n_p_min = 30
        n_p_max = n_0 + n_slope
        n_p = np.arange(n_p_min, n_p_max, 15)

        inflow = np.arange(min(data_inflows), max(data_inflows), 1)
        n_p, inflow = np.meshgrid(n_p, inflow)
        T = (CT[0] + CT[1] * inflow / (n_p * D_p))


        norm = plt.Normalize(T.min(),T.max())
        colors = cm.viridis(norm(T))
        rcount, ccount, _ = colors.shape




        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=50., azim=-75)
        surf = ax.plot_surface(n_p, inflow, T, rcount=rcount, ccount=ccount, facecolors=colors, label='Datapoints', shade= False)
        surf.set_facecolor((0, 0, 0, 0))
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'), shrink=0.4, pad=0.1)
        norm = plt.Normalize(T.min(), T.max())
        colors = cm.viridis(norm(thrust_datapoints))
        rcount, ccount, _ = colors.shape

        ax.scatter(n_ps, data_inflows, thrust_datapoints, c=thrust_datapoints, cmap = 'viridis', norm = norm, zdir='z', s=10, label='Least-squares fit', depthshade= False)
        if show_titles: ax.set_title(f'Identified Thrust Coefficients')
        ax.set_xlabel('\n'+r'Propeller speed $n_p$ $[\frac{rot}{s}]$')
        ax.set_ylabel('\n'+r'Inflow  $v_i$ $[\frac{m}{s}]$')
        ax.set_zlabel('\n'+r'Combined thrust coefficient $[1]$')
        # ax.set_zlim(-0.3, 0.1)
        plt.tight_layout()
        plt.show()

    return CT

if __name__ == '__main__':
    CE = calc_elev(_model_name, verbose=_verbose, show_plots=_show_plots)
    CL, CD = calc_lift_drag(_model_name, _fix_cd2, verbose=_verbose, show_plots=_show_plots, env=_env)
    calc_thrust(_model_name, CL, CD, verbose=_verbose, show_plots=_show_plots, env=_env)


