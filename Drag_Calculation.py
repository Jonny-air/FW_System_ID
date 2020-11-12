# params
_rho = 1.225

def get_drag(CD, vas, alphas, rho = _rho):
    D = 0.5 * rho * vas ** 2 * (CD[0] + CD[1] * alphas + CD[2] * alphas ** 2)
    return D