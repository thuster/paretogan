import torch
import scipy.stats as stats
import numpy as np
import os
from zipfile import ZipFile
import gzip
import csv
from tqdm import tqdm
import pandas as pd

from os import listdir
from os.path import isfile, join

class DS(torch.utils.data.Dataset):
    # Dataset class for pytorch
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]

    def merge(self, datasets):
        if type(datasets) is DS:
            datasets = [datasets]
        self.X = torch.cat([self.X] + [d.X for d in datasets], dim=0)

    def pdf(self, x):
        bin_sz = x[1] - x[0]
        bins = np.concatenate([x, [x[-1] + bin_sz]], axis=0)
        d, b = np.histogram(self.X.numpy().reshape(-1), bins=bins, density=True)
        return d


class Stat_DS(DS):
    # Special class for datasets with known distributions
    def __init__(self, dist=stats.cauchy, dsize=(10000, 1), seed=None, **kwargs):
        self.rv = dist(**kwargs)
        if seed is not None:
            self.rv.random_state = np.random.RandomState(seed=seed)

        X = self.rv.rvs(size=dsize)
        X = torch.Tensor(X).float()
        super().__init__(X)
        self.pdf = lambda x: self.rv.pdf(x)
        self.cdf = lambda x: self.rv.cdf(x)


    def merge(self, datasets):
        if type(datasets) is not list:
            datasets = [datasets]
        datasets.append(self)

        pdfs = [d.pdf for d in datasets]
        cdfs = [d.cdf for d in datasets]
        weights = [d.X.shape[0] for d in datasets]

        self.pdf = lambda x: np.sum([w*f(x) for f, w in zip(pdfs, weights)], axis=0)/sum(weights)
        self.cdf = lambda x: np.sum([w * f(x) for f, w in zip(cdfs, weights)], axis=0) / sum(weights)

        self.X = torch.cat([d.X for d in datasets], dim=0)


def cauchy_ds():
    return [Stat_DS(seed=0), Stat_DS(seed=1)]


def dual_cauchy_ds(test=False, seed=0):
    a0 = Stat_DS(dsize=(7000, 1), seed=seed)
    b0 = Stat_DS(dsize=(3000, 1), seed=seed, loc=3, scale=0.5)
    a0.merge(b0)

    a1 = Stat_DS(dsize=(7000, 1), seed=seed+1)
    b1 = Stat_DS(dsize=(3000, 1), seed=seed+1, loc=3, scale=0.5)
    a1.merge(b1)

    if test:
        a2 = Stat_DS(dsize=(7000000, 1), seed=seed+2)
        b2 = Stat_DS(dsize=(3000000, 1), seed=seed+2, loc=3, scale=0.5)
        a2.merge(b2)
        return [a0, a1, a2]

    return [a0, a1]




def wiki_web_traffic_dataset(folder=os.path.join('data', 'web'),
                             class_index=2,
                             length=10,
                             scale=1.0,
                             training_test_ratio=0.5,
                             random_seed=0,
                             addnoise=True):
    """Load wiki web traffic dataset.
    https://www.kaggle.com/c/web-traffic-time-series-forecasting/data

    Args:
        folder (str): The folder for storing data files.
        class_index (int): The ID of the class label. The values can be:
            0: The name of page.
            1: The domain name of the page: ['commons.wikimedia.org',
            'de.wikipedia.org', 'en.wikipedia.org', 'es.wikipedia.org',
            'fr.wikipedia.org', 'ja.wikipedia.org', 'ru.wikipedia.org',
            'www.mediawiki.org', 'zh.wikipedia.org']
            2: The access type: ['all-access', 'desktop', 'mobile-web']
            DEPRECATED 3: The agent type: ['all-agents', 'spider']
        length (int): Timeseries cut length.
        scale (float): The scale multiplied to timeseries values.
        training_test_ratio (float): The ratio between number of training and
            test samples.
        random_seed (int): The random seed for data sample shuffling.

    Returns:
        [DS]: Training datasets. A list of DS objects, one for each class.
        [DS]: Test datasets. A list of DS objects, one for each class.

    """
    csv_path = os.path.join(folder, 'train_1.csv')

    if not os.path.exists(csv_path):
        # download and extract datasets

        all_zip_path = os.path.join(
            folder, 'web-traffic-time-series-forecasting.zip')
        if not os.path.exists(all_zip_path):
            raise Exception(
                'Please download web-traffic-time-series-forecasting.zip from '
                'https://www.kaggle.com/c/web-traffic-time-series-forecasting/'
                'data and save it to folder: {}'.format(folder))

        print('Extracting files')
        with ZipFile(all_zip_path, 'r') as zip_obj:
            zip_obj.extract('train_1.csv.zip', folder)

        csv_zip_path = os.path.join(folder, 'train_1.csv.zip')
        with ZipFile(csv_zip_path, 'r') as zip_obj:
            zip_obj.extract('train_1.csv', folder)

        assert os.path.exists(csv_path)

    print('Reading data')

    def __get_keys(key):
        key = key.split('_')
        return ['_'.join(key[:-3])] + key[-3:]

    timeseries = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in tqdm(reader):
            s_key = __get_keys(row[0])[class_index]
            if class_index == 2 and s_key == 'all-access':
                continue
            s_timeseries = row[1:]

            # ignore rows with missing data
            if '' in s_timeseries:
                continue

            s_timeseries = list(map(float, s_timeseries))
            if s_key not in timeseries:
                timeseries[s_key] = []
            timeseries[s_key].append(s_timeseries[:length])

    print('Found {} classes: {}'.format(
        len(timeseries),
        list(timeseries.keys())))
    total_num_samples = 0
    for k in timeseries:
        num_samples = len(timeseries[k])
        print('Class {} has {} samples'.format(k, num_samples))
        total_num_samples += num_samples
    print('Total number of samples: {}'.format(total_num_samples))

    training_datasets = []
    test_datasets = []
    random_state = np.random.RandomState(random_seed)

    for k in timeseries:
        dataset = np.asarray(timeseries[k]) * scale
        if addnoise:
            dataset += random_state.rand(*dataset.shape) * scale

        random_state.shuffle(dataset)

        split = int(dataset.shape[0] * training_test_ratio)
        training_dataset = DS(torch.Tensor(dataset[:split]).float())
        test_dataset = DS(torch.Tensor(dataset[split:]).float())

        training_datasets.append(training_dataset)
        test_datasets.append(test_dataset)

    return training_datasets, test_datasets



def wiki_1Dcomb(seed=0):


    training_datasets, test_datasets = wiki_web_traffic_dataset(folder=os.path.join('data', 'web'),
                             class_index=2,
                             length=1,
                             scale=1.0,
                             training_test_ratio=0.2,
                             random_seed=seed)


    tr1 = training_datasets[0].X
    split1 = round(tr1.shape[0]/2)

    tr2 = training_datasets[1].X
    split2 = round(tr2.shape[0] / 2)

    training = DS(tr1[:split1])
    val = DS(tr1[split1:])

    training.merge(DS(tr2[:split2]))
    val.merge(DS(tr2[split2:]))

    testing = test_datasets[0]
    testing.merge(training_datasets[1])

    return training, val, testing


def key_preproc():

    folder = 'data/136Mkeystrokes/'
    subdir = 'Keystrokes/files'
    files_path = os.path.join(folder, subdir)


    if not os.path.exists(files_path):
        # download and extract dataset

        all_zip_path = os.path.join(
            folder, 'Keystrokes.zip')
        if not os.path.exists(all_zip_path):
            raise Exception(
                'Please download Keystrokes.zip from '
                'https://userinterfaces.aalto.fi/136Mkeystrokes/data/Keystrokes.zip'
                'data and save it to folder: {}'.format(folder))


        print('Extracting files - this takes a few minutes')

        import zipfile
        with zipfile.ZipFile(all_zip_path, 'r') as zip_ref:
            zip_ref.extractall(folder)

        assert os.path.exists(files_path)
        print('Files extracted')

    print('Processing files - this takes a few minutes')
    fns = [f for f in listdir(files_path) if isfile(join(files_path, f))]

    iatimes = []

    for fn in fns:
        try:
            table = pd.read_csv(join(files_path, fn), sep='\t')

            times = table['PRESS_TIME'].to_numpy()
            durations = times[1:] - times[:-1]

            iatimes.append(durations)
        except:
            print("pandas couldn't read ",fn)


    iatimes = np.concatenate(iatimes)
    iatimes = iatimes[~np.isnan(iatimes)]

    outfn = 'keystrokes.npy'

    np.save(join(folder, outfn), iatimes)



def keystroke_ds(seed=0, tr_sz=1000000):

    basedir = 'data/136Mkeystrokes'
    fn = 'keystrokes.npy'

    if not os.path.exists(join(basedir, fn)):
        key_preproc()


    data = np.load(join(basedir, fn))
    data = data[data>0]

    rng = np.random.RandomState(seed=seed)
    rng.shuffle(data)
    data = torch.Tensor(data).float().reshape(-1,1)

    tr = DS(data[:tr_sz])
    v = DS(data[tr_sz:2*tr_sz])
    tst = DS(data[2*tr_sz:])

    return tr, v, tst



def livej_preproc():
    folder = 'data/livej/'
    fn = 'soc-LiveJournal1.txt'
    txt_path = os.path.join(folder, fn)

    if not os.path.exists(txt_path):
        # download and extract dataset

        all_zip_path = os.path.join(
            folder, 'soc-LiveJournal1.txt.gz')
        if not os.path.exists(all_zip_path):
            raise Exception(
                'Please download soc-LiveJournal1.txt.gz from '
                'https://snap.stanford.edu/data/soc-LiveJournal1.html'
                'data and save it to folder: {}'.format(folder))

        print('Extracting files')

        input = gzip.GzipFile(all_zip_path, 'rb')
        s = input.read()
        input.close()

        output = open(txt_path, 'wb')
        output.write(s)
        output.close()


        assert os.path.exists(txt_path)


    nodes = pd.read_csv(join(folder, fn), sep='\t', comment='#').to_numpy().reshape(-1)
    (node, edgecount) = np.unique(nodes, return_counts=True)
    outfn = 'livej.npy'

    np.save(join(folder, outfn), edgecount)


def livej_ds(seed=0, tr_sz=10000):
    basedir = 'data/livej'
    fn = 'livej.npy'

    if not os.path.exists(join(basedir, fn)):
        livej_preproc()

    data = np.load(join(basedir, fn))

    rng = np.random.RandomState(seed=seed)
    rng.shuffle(data)
    data = torch.Tensor(data).float().reshape(-1,1)

    tr = DS(data[:tr_sz])
    v = DS(data[tr_sz:2 * tr_sz])
    tst = DS(data[2 * tr_sz:])

    return tr, v, tst

