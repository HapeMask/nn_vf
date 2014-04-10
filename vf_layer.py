import numpy as np
import theano.tensor as tt
from scipy.optimize import fmin_l_bfgs_b

def apply_vf(field_pts, field_vecs, pts, sigma=0.25):
    dists = ((pts[:, np.newaxis, :] - field_pts[np.newaxis, :, :])**2).sum(-1)
    weights = tt.exp(-dists/(2*sigma**2))
    displacements = (field_vecs[np.newaxis, :, :] * weights[:, :, np.newaxis]).sum(1)

    return pts + displacements

def optimize_nn(sym_params, f_and_g, max_fun_calls=200, verbose=False):
    x0 = 0.1*(np.random.normal(size=sym_params.get_value().shape))
    x_opt = fmin_l_bfgs_b(
                    func    = f_and_g,
                    x0      = x0,
                    maxfun  = max_fun_calls,
                    m       = 32,
                    disp    = 10 if verbose else None)[0]
    return x_opt
