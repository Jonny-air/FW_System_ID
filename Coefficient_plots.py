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
    vas, va_dots, fpas, fpa_dots, alphas, rolls, pitches, elevs = import_data(model_name, 0)
    g, rho = env
    if params is None:
        try:
            m = get_params(model_name)[0]
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

    fig, axs = plt.subplots(2, figsize=(7, 7))
    ax1 = axs[0]
    ax2 = axs[1]


    ax1.plot(alpha * 180 / np.pi, L, 'g', label='Lift Coefficient')
    ax1.plot(alphas * 180 / np.pi, lift_datapoints, 'x', label='Lift Datapoints')
    ax2.plot(alpha * 180 / np.pi, D, 'g', label='Drag Coefficient')
    ax2.plot(alphas * 180 / np.pi, drag_datapoints, 'x', label='Drag Datapoints')

    liftstr = '\n'.join((
        r'$C_{L_0}=%.2f$' % (CL[0],),
        r'$C_{L_\alpha}=%.2f$' % (CL[1],)))
    if resL is not None:
        liftstr = '\n'.join((liftstr, r'Residual=%.2f' % (resL,)))

    dragstr = '\n'.join((
        r'$C_{D_0}=%.2f$' % (CD[0],),
        r'$C_{D_\alpha}=%.2f$' % (CD[1],)))
    if fix_cd2 is True:
        dragstr = '\n'.join((dragstr, r'Fixed $C_{D_{\alpha^2}}=%.2f$' % (CD[2],)))
    else:
        dragstr = '\n'.join((dragstr, r'$C_{D_{\alpha^2}}=%.2f$' % (CD[2],)))
    if resL is not None:
        dragstr = '\n'.join((dragstr, r'Residual=%.2f' % (resD,)))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='square', facecolor='whitesmoke', alpha=0.7)

    # place a text box in upper left in axes coords
    ax1.text(0.95, 0.1, liftstr, transform=ax1.transAxes,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    ax2.text(0.95, 0.1, dragstr, transform=ax2.transAxes,
             verticalalignment='bottom', horizontalalignment='right', bbox=props)

    ax1.set_title(f'Identified Lift and Drag Coefficients')
    ax1.set_xlabel(r'Angle of attack: $\alpha$ [deg]')
    ax2.set_xlabel(r'Angle of attack: $\alpha$ [deg]')
    ax1.set_ylabel(r'Lift Coefficient [$m^2$]')
    ax2.set_ylabel(r'Drag Coefficient [$m^2$]')
    ax1.legend(loc='best')
    ax2.legend(loc='best')
    ax1.grid()
    ax2.grid()
    plt.show()

if __name__ == '__main__':
    plot_lift_drag(_CL, _CD, get_datapoints(_model_name))