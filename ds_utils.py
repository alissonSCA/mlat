import numpy as np
import pandas as pd
from rssi import RSSI_Localizer
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.stats import nakagami
import pickle


def plot_data(data, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1)
    for i in np.arange(data['K']):
        # ax.scatter(data['R'][i][0], data['R'][i][1], color='C%d' % (i))
        for j in np.arange(data['N'][i]):
            c = plt.Circle(data['R'][i], data['delta'][i, j], fill=False, color='C%d' % i, alpha=0.2)
            ax.add_patch(c)

    ax.plot(data['t'][0][0], data['t'][0][1], 'k*')
    return ax


def generate_artificial_data(k, n, radius, ref_p_noise_std, theta, q_std=10):
    Alpha = [(2 * np.pi) * (i / k) for i in 1 + np.arange(k)]
    noi = np.random.multivariate_normal([0, 0], np.eye(2) * (q_std ** 2), size=1)
    R = np.array([noi + [radius * np.cos(alpha), radius * np.sin(alpha)] for alpha in Alpha])[:, 0, :]

    naka = lambda d, theta: nakagami(nu=(theta + d ** 2) / (2 * theta), scale=np.sqrt(theta + d ** 2)).rvs(1)[0]

    D = np.zeros([k, n])
    for i in np.arange(k):
        R_sample = np.random.multivariate_normal(R[i, :], np.eye(2) * ref_p_noise_std ** 2, n)
        x = distance.cdist(R_sample, noi)
        D[i] = [naka(xi, theta[i]) for xi in x[:, 0]]

    return {'t': noi, 'R': R, 'delta': D, 'K': k, 'N': int(n)*np.ones(k, dtype=np.int), 'N_max': n, 'D': 2, 'sigma_r': ref_p_noise_std}


def get_reference(df, R, noi):
    distances = distance.cdist([noi], R)[0, :]
    signals = df.groupby(['ap']).mean()['signal']
    reference = []
    for d, s in zip(distances, signals):
        reference.append({'distance': d, 'signal': -s})
    return reference


def get_distance_from_rssi(R, reference, signalStrength):
    d = RSSI_Localizer.getDistanceFromAP({'signalAttenuation': 3,
                                          'location': {'x': R[0], 'y': R[1]},
                                          'reference': reference,
                                          'name': 'dd-wrt'}, -signalStrength)
    return d['distance']


def generate_rssi_ds(x, y):
    # dataset from https://www.kaggle.com/amirma/indoor-location-determination-with-rssi
    R = {"A": [23, 17, 2], "B": [23, 41, 2], "C": [1, 15, 2], "D": [1, 41, 2]}
    df = pd.read_csv('./data/rssi.csv')
    df = df[df['x'] == x]
    df = df[df['y'] == y]
    df = df[df['z'] == 2]
    data = {'q': np.array([[x, y]])}

    get_ref_p_index = lambda x: 'ABCD'.find(x)
    ref_p_dir = {0: [], 1: [], 2: [], 3: []}
    for i, dt in df.iterrows():
        ref_p_dir[get_ref_p_index(dt['ap'])].append(dt['signal'])

    data['n'] = [len(ref_p_dir[x]) for x in ref_p_dir]
    data['n_max'] = np.max(data['n'])
    data['k'] = 4
    data['dim'] = 2
    data['R'] = np.array([[23, 17], [23, 41], [1, 15], [1, 41]])

    reference = get_reference(df, data['R'], [x, y])
    D = np.zeros([4, data['n_max']])
    for i in np.arange(4):
        Ri = data['R'][i, :]
        ind = np.random.permutation(len(ref_p_dir[i]))
        for j in np.arange(data['n'][i]):
            D[i, j] = get_distance_from_rssi(Ri, reference[i], ref_p_dir[i][ind[j]])
    data['D'] = D
    return data
