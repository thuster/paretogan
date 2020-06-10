import numpy as np
import tail_estimation
from scipy.stats import genpareto
import torch

def fit_tail(data, verbose=False, tail_ratio=0.05):
    data = data.reshape(-1)
    data = data[:round(data.shape[0] * tail_ratio)]
    shape, location, scale = genpareto.fit(data)
    if verbose:
        print(f"Shape parameter: {shape}\nLocation: {location}\nScale: {scale}")
    return shape


def get_tail_shape(data, methods=('kernel','moments','hill')):
    data = data.numpy()
    data = data.reshape(-1)
    data = np.abs(data)

    data[::-1].sort()

    hsteps = 200
    eps_stop = 1 - float(len(data[np.where(data <= 0)])) / len(data)
    shapes=[]

    if 'hill' in methods:
        hill_results = tail_estimation.hill_estimator(data, eps_stop=eps_stop)
        hill_xi_star = hill_results[3]
        print('hill:', hill_xi_star)
        shapes.append(hill_xi_star)

    if 'moments' in methods:
        moments_results = tail_estimation.moments_estimator(data, eps_stop=eps_stop)
        moments_xi_star = moments_results[3]
        print('moments:', moments_xi_star)
        shapes.append(moments_xi_star)

    if 'kernel' in methods:
        kernel_type_results = tail_estimation.kernel_type_estimator(data, hsteps, eps_stop=eps_stop)
        kernel_type_xi_star = kernel_type_results[3]
        print('kernel:', kernel_type_xi_star)
        shapes.append(kernel_type_xi_star)

    estimate = np.mean(shapes)
    estimate = np.maximum(0, estimate)

    return estimate



def get_noise_fn(data, shape=None, in_size=20):
    if shape is None:
        shape = get_tail_shape(data)
    dist = genpareto(shape)
    noise_fn = lambda bsz: torch.Tensor(dist.rvs(size=[bsz, in_size]))
    return noise_fn, shape






