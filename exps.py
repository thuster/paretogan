import torch
import numpy as np
import matplotlib.pyplot as plt

import utils
import datasets
import models
import tail_shape
import mmd


archive_path = 'data/samples/'

test_datasets = [
    lambda: datasets.keystroke_ds(seed=0),
    lambda: datasets.wiki_1Dcomb(seed=0),
    lambda: datasets.livej_ds(seed=0,tr_sz=50000),
    lambda: datasets.dual_cauchy_ds(test=True, seed=0)
]

def exp(dsind, alg='pareto'):

    lr = 0.001
    in_size = 2

    h_size = 32
    iters = 20000

    assert dsind<len(test_datasets), "dataset index must be less than " + str(len(test_datasets))
    ds = test_datasets[dsind]()

    mean = (np.abs(ds[0].X)**(0.5)).mean().numpy()

    ds[0].X = ds[0].X / mean
    ds[1].X = ds[1].X / mean
    ds[2].X = ds[2].X / mean

    if alg == 'pareto':
        shape = tail_shape.get_tail_shape(ds[0].X, methods=('kernel',))
        model = models.ParetoGenStatic(in_size, h_size, 1, alpha=1, xi=shape)
        noise_fn = lambda bsz: (1 - torch.rand(bsz, in_size)) ** -1

        gamma = 2
        loss_fn = lambda X, Y: mmd.ed_kernel_poly(X, Y, gamma=gamma)
        ks, area_dist = run_exp(ds, model, noise_fn, loss_fn, iters=iters, lr=lr)

    elif alg == 'uniform':
        model = models.Generator(in_size, h_size, 1)
        noise_fn = lambda bsz: torch.rand(bsz, in_size) - 0.5

        gamma = 1
        loss_fn = lambda X, Y: mmd.ed_kernel_poly(X, Y, gamma=gamma)
        ks, area_dist = run_exp(ds, model, noise_fn, loss_fn, iters=iters, lr=lr)

    elif alg == 'normal':
        model = models.Generator(in_size, h_size, 1)
        noise_fn = lambda bsz: torch.randn(bsz, in_size)

        gamma = 1
        loss_fn = lambda X, Y: mmd.ed_kernel_poly(X, Y, gamma=gamma)
        ks, area_dist = run_exp(ds, model, noise_fn, loss_fn, iters=iters, lr=lr)

    elif alg == 'lognormal':

        new_ds = datasets.DS(utils.logtransform(ds[0].X))
        ln_ds = [new_ds] + list(ds[1:])


        model = models.Generator(in_size, h_size, 1)
        noise_fn = lambda bsz: torch.randn(bsz, in_size)

        gamma = 1
        loss_fn = lambda X, Y: mmd.ed_kernel_poly(X, Y, gamma=gamma)
        ks, area_dist = run_exp(ln_ds, model, noise_fn, loss_fn, iters=iters, lr=lr,
                       output_transform=utils.loguntransform)
    else:
        raise ValueError("alg must be one of 'pareto','uniform','normal','lognormal'")

    return ks, area_dist


def run_exp(dataset, model, noise_fn, loss_fn, lr=1e-4, iters=10000, output_transform=None, expname=''):
    train_ds = dataset[0]
    val_ds = dataset[1]
    test_ds = dataset[2]

    utils.train(model, noise_fn, loss_fn, train_ds, val_ds, iters=iters, lr=lr)

    # vdist = evaluate(model, noise_fn, loss_fn, val_ds, output_transform=output_transform,expname=expname)
    ks, area_dist = evaluate(model, noise_fn, loss_fn, test_ds, output_transform=output_transform,expname=expname)

    return ks, area_dist

from scipy.stats import ks_2samp
def evaluate(model, noise_fn, lossfn, ds, output_transform=None,expname=''):

    real = ds.X

    nsamples = len(real)
    noise = noise_fn(nsamples)

    # model.cpu()

    batch_sz = 2000
    nbatches = int(np.ceil(nsamples / batch_sz))

    fakes = []
    for i in range(nbatches):
        # noise[batch_sz*i:batch_sz*(i+1)]
        fake = model(noise[batch_sz * i:batch_sz * (i + 1)].cuda()).detach().cpu().numpy().reshape(-1)
        # fake = model(noise[batch_sz*i:batch_sz*(i+1)]).detach().numpy().reshape(-1)
        if output_transform is not None:
            fake = output_transform(fake)
        fakes.append(fake)

    print('made fakes')

    fake = np.concatenate(fakes)


    real = real.detach().numpy().reshape(-1)
    batch_sz = np.minimum(2000,nsamples)

    nbatches = int(np.floor(nsamples / batch_sz))
    nbatches = np.minimum(20,nbatches)

    lss = 0
    for i in range(nbatches):
        fake_batch = torch.Tensor(fake[batch_sz*i:batch_sz*(i+1)]).reshape(-1,1).cuda()
        real_batch = torch.Tensor(real[batch_sz*i:batch_sz*(i+1)]).reshape(-1,1).cuda()

        lss += lossfn(fake_batch, real_batch)/nbatches

    lss = lss.cpu().detach().numpy()

    print('computed loss')
    ks = ks_2samp(fake, real)[0]
    # area_dist = area_metric(fake, real)
    area_dist = area(fake, real)

    print('ks statistic:', ks)
    print('log-log area:', area_dist)

    loglogplot(real, fake)
    histplot(real, fake)

    return ks, area_dist


def area(real, fake):
    assert len(real)==len(fake)

    z = 1 - np.arange(len(real)) / float(len(real))

    fake = np.maximum(1e-6, fake)
    real = np.maximum(1e-6, real)

    fake.sort()
    real.sort()


    real = np.log(real)
    fake = np.log(fake)
    z = np.log(z)


    diffs = np.abs(real-fake)

    widths = z[:-1]-z[1:]


    wdiffs = widths*diffs[1:]
    # return wdiffs[:-1].sum()
    return wdiffs.sum()


def loglogplot(real, fake):


    real.sort()
    fake.sort()

    real_s = 1 - np.arange(len(real))/float(len(real))

    plt.figure()

    plt.plot(real, real_s)
    plt.plot(fake, real_s)

    plt.xlim(left=1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('x (normalized)')
    plt.ylabel('P(X>x)')
    plt.legend(['Real', 'Generated'], loc='upper right')

    plt.show()


def histplot(real, fake):
    percentile = .02
    real.sort()
    lb = real[round(percentile*len(real))]
    ub = real[round((1-percentile) * len(real))]
    bins = np.linspace(lb, ub, 200)


    plt.figure()
    plt.hist(fake, bins=bins, density=True)
    n, x = np.histogram(real, bins=bins, density=True)
    x = (bins[1:]+bins[:-1])/2
    plt.plot(x, n, linewidth=3)
    plt.xlabel('x (normalized)')
    plt.ylabel('P(x)')
    plt.legend(['Real','Generated'], loc='upper right')
    # plt.legend(['Generated', 'Real'], loc='upper right')

    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a GAN on a heavy tailed dataset')

    parser.add_argument('-type', action="store", default='pareto')
    parser.add_argument('-ds', action="store", type=int, default=3)


    args = parser.parse_args()

    exp(args.ds, alg=args.type)