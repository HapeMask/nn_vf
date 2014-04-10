import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as tt

from vf_layer import *

def plot_probabilities(sym_prob, sym_input, rng):
    start, end = rng
    xg, yg = np.mgrid[start:end:128j, start:end:128j].astype(np.float32)
    grid = np.dstack([xg, yg]).reshape(-1, 2)

    probs = sym_prob.eval({sym_input:grid}).reshape(128,128,2)[:,:,1]

    plt.contourf(xg, yg, probs, cmap="RdBu")
    plt.plot(X[y==0, 0], X[y==0, 1], 'r.')
    plt.plot(X[y==1, 0], X[y==1, 1], 'b.')

n_samples = 2048

n_in = 2
n_classes = 2
n_field_pts = 32*32
L1_penalty_weight = 1e-1

# Generate a toy 2-class dataset with a donut of points and a set of center
# points.
pts = np.random.normal(size=(n_samples, 2))
r = np.sqrt((pts**2).sum(-1))
p1 = pts[r < 0.5] * 0.25
p2 = pts[r > 0.5]
X = np.vstack([p1, p2]).astype(np.float32)
y = np.hstack([np.zeros((p1.shape[0],)), np.ones((p2.shape[0],))]).astype(np.int32)

Xs = tt.matrix("Xs")
ys = tt.ivector("ys")

# Allocate space for parameters and split the vector into the individual
# parameter groups.
params = theano.shared(np.zeros((2*n_field_pts*n_in + n_in*n_classes + n_classes,), np.float32), "params")
field_pts = params[:n_field_pts*n_in].reshape((n_field_pts, n_in))
field_vecs = params[n_field_pts*n_in:2*n_field_pts*n_in].reshape((n_field_pts, n_in))
W = params[2*n_field_pts*n_in:2*n_field_pts*n_in + n_in*n_classes].reshape((n_in, n_classes))
b = params[2*n_field_pts*n_in + n_in*n_classes:][np.newaxis,:]

# Apply the vector field layer to the input, then apply a softmax layer to the
# transformed points.
transformed = apply_vf(field_pts, field_vecs, Xs)
probs = tt.nnet.softmax(tt.dot(transformed, W) + b)

nll = -tt.log(probs[tt.arange(ys.shape[0]), ys]).mean()

# NLL + L1-penalty on field vector magnitudes.
objective = nll + L1_penalty_weight*abs(field_vecs).sum(-1).mean()

obj_and_grad = theano.function([Xs, ys], [objective, tt.grad(objective, params)])
predict = theano.function([Xs], tt.argmax(probs, axis=1))

def f_and_g(x):
    global X, y

    params.set_value(x.astype(np.float32))
    f, g = obj_and_grad(X, y)
    return f, g.astype(np.float64)

opt_params = optimize_nn(params, f_and_g, verbose=True)
params.set_value(opt_params.astype(np.float32))

#
# Visualize results.
#
plot_probabilities(probs, Xs, (X.min(), X.max()))
plt.savefig("probability_map.png")
plt.close()

fp = field_pts.eval()
fv = field_vecs.eval()
t = transformed.eval({Xs:X})

plt.quiver(fp[:, 0], fp[:, 1], fv[:, 0], fv[:, 1])
plt.xlim(t[:,0].min(), t[:,0].max())
plt.ylim(t[:,1].min(), t[:,1].max())
plt.savefig("vector_field.png")
plt.close()

plt.plot(t[y==0, 0], t[y==0, 1], 'r.')
plt.plot(t[y==1, 0], t[y==1, 1], 'b.')
plt.xlim(t[:,0].min(), t[:,0].max())
plt.ylim(t[:,1].min(), t[:,1].max())
plt.savefig("transformed_points.png")
plt.close()
