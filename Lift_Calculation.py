# params
_rho = 1.225

def get_lift(CL, vas, alphas, rho = _rho):
    L = 0.5 * rho * vas ** 2 * (CL[0] + CL[1] * alphas)
    return L