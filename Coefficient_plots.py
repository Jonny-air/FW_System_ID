import numpy as np
import matplotlib.pyplot as plt
from Import_Datapoints import import_data
from load_model import get_params

# Parameters for manual plot
_model_name = "Believer_201112_true"
_g = 9.80
_rho = 1.225
_CL = [0.1302, 1.816]
_CD = [0.0256, 0.1553, 0.1204]

def get_datapoints(model_name, params = None, env = [_g, _rho]):
    vas, va_dots, fpas, fpa_dots, alphas, rolls = import_data(model_name, 0)
    g, rho = env
    if params is None:
        try:
            [m, n_0, n_slope, thrust_incl, D_p, actuator_control_ch, state_level] = get_params(model_name)
        except Exception as e:
            print("No Parameters found for the datapoints")
            print(e)
            return 0
    lift_datapoints = 2 * m / (rho * vas ** 2) * (fpa_dots + g * np.cos(fpas))
    drag_datapoints = -2 * m / (rho * (vas ** 2)) * (va_dots + g * np.sin(fpas))
    return [alphas, lift_datapoints, drag_datapoints]


def plot_lift_drag(CL, CD, datapoints, resL = None, resD = None, fix_cd2 = None):
    alpha_start = -7 * np.pi / 180
    alpha_end = 10 * np.pi / 180
    alpha_step = 0.1 * np.pi / 180
    alpha = np.arange(alpha_start, alpha_end, alpha_step)
    L = (CL[0] + CL[1] * alpha)
    D = (CD[0] + CD[1] * alpha + CD[2] * alpha ** 2)
    alphas, lift_datapoints, drag_datapoints = datapoints

    fig, axs = plt.subplots(2, figsize=(10, 10))
    ax1 = axs[0]
    ax2 = axs[1]


    ax1.plot(alpha * 180 / np.pi, L, 'g', label='Lift')
    ax1.plot(alphas * 180 / np.pi, lift_datapoints, 'x', label='Lift Datapoints')
    ax2.plot(alpha * 180 / np.pi, D, 'g', label='Drag')
    ax2.plot(alphas * 180 / np.pi, drag_datapoints, 'x', label='Drag Datapoints')
    if resL is not None:
        plt.figtext(.7, .75, f"Residual = {round(resL, 5)}")
    if resD is not None:
        plt.figtext(.7, .26, f"Residual = {round(resD, 5)}")
    if fix_cd2 is None:
        ax1.set_title("Lift and Drag Coefficients with datapoints from ID")
    else:
        plt.figtext(0.3, .9, f"Coefficients with CD[2] fixed = {fix_cd2}:\nCD: [{CD[0]}, {CD[1]}, {CD[2]}]\nCL: [{CL[0]}, {CL[1]}]")

    ax1.set_xlabel('Angle of Attack in Degrees')
    ax2.set_xlabel('Angle of Attack in Degrees')
    ax1.set_ylabel('Lift Coefficients [m^2]')
    ax2.set_ylabel('Drag Coefficients [m^2]')
    ax1.legend(loc='best')
    ax2.legend(loc='best')
    ax1.grid()
    ax2.grid()
    plt.show()

if __name__ == '__main__':
    plot_lift_drag(_CL, _CD, get_datapoints(_model_name))