import numpy as np
# params
_rho = 1.225

def get_thrust(CT, vas, alphas, n_ps, thrust_incl, D_p, rho = _rho):
    T = rho * n_ps ** 2 * D_p ** 4 * (CT[0] + CT[1] * vas * np.divide(np.cos(alphas - thrust_incl), (n_ps * D_p), out=np.zeros_like(n_ps), where=n_ps!=0))
    return T