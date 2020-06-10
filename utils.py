
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt


def logtransform(x):
    y = np.sign(x)*np.log(np.abs(x)+1)
    return y


def loguntransform(x):
    y = np.sign(x)*(np.exp(np.abs(x))-1)
    return y



def train(model, noise_fn, loss_fn, train_ds, test_ds, iters=10000, lr=1e-4, batch_size=256):

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model = model.to(device)

    optimizerG = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.9))
    schedG = optim.lr_scheduler.ExponentialLR(optimizerG, 0.985, last_epoch=-1)

    for iteration in range(iters):

        model.zero_grad()
        batch = np.random.randint(0, len(train_ds), size=batch_size)

        real = train_ds[batch].to(device)

        noise = noise_fn(batch_size)
        noise = torch.Tensor(noise)
        noise = noise.to(device)

        fake = model(noise)

        loss = loss_fn(fake, real)
        loss.backward()
        optimizerG.step()

        # Write logs and save samples
        if (iteration + 1) % 200 == 0:
            print('Iteration', iteration + 1,
                  'Learning rate:', schedG.get_lr(),
                  'mmd:', loss.item())
            schedG.step()


def compare(model, noise_fn, pdf_fun=None, tgt_data=None, model_path=None, output_transform=None, std=True, modelims=(-10,10),taillims=(-100,100)):
    nsamples = 1000000
    noise = noise_fn(nsamples)

    model.cpu()
    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

    # lc = quicklc(model)
    # print('Lipschitz Constant:', lc)

    fake = model(noise).detach().numpy()
    if output_transform is not None:
        fake = output_transform(fake)
    else:
        fake = fake

    bins_mode = np.linspace(modelims[0], modelims[1], 100)

    # fake = fake.cpu().detach().numpy()

    # bins_tails = np.linspace(fake.min(), fake.max(), 100)
    bins_tails = np.linspace(taillims[0], taillims[1], 100)
    bins_mode_pdf = np.linspace(modelims[0], modelims[1], 100)
    bins_tails_pdf = np.linspace(taillims[0], taillims[1], 100)

    if pdf_fun is not None:
        tgt_mode = pdf_fun(bins_mode_pdf).reshape(-1, 1)
        tgt_tails = pdf_fun(bins_tails_pdf).reshape(-1, 1)
    else:
        tgt_mode = None
        tgt_tails = None

    plt.figure()
    plt.hist(fake, bins=bins_mode, density=True)
    plt.plot(bins_mode_pdf, tgt_mode)
    # plt.legend(['Target distribution', 'Generated data'])
    plt.xlabel('x')
    plt.ylabel('Probaility Density')
    plt.show()


    plt.figure()
    plt.hist(fake, bins=bins_tails, density=True)
    plt.plot(bins_tails_pdf, tgt_tails)
    # plt.legend(['Target distribution', 'Generated data'])
    plt.xlabel('x')
    plt.ylabel('Probaility Density')
    plt.yscale('log')
    plt.show()


