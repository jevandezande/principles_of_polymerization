import numpy as np

from mw import averages, weight_fractions
from tools import flory_schulz

from matplotlib import pyplot as plt
from scipy.signal import general_gaussian


def graph_wf_vs_nf(numbers, weights, mw_averages=True):
    """
    Graph how weight fraction changes distribution shape.
    :param numbers: number of particles
    :param weights: weight of particles
    :return: fig, ax
    """
    numbers, weights = np.asarray(numbers), np.asarray(weights)

    wf = weight_fractions(numbers, weights)
    nf = numbers/sum(numbers)
    y_max = max(max(nf), max(wf))

    fig, ax = plt.subplots()
    ax.plot(weights, nf, label='Number Fraction')
    ax.plot(weights, wf, label='Weight Fraction')

    # M numbers
    if mw_averages:
        if mw_averages is True:
            for name, average_function in averages.items():
                w = average_function(numbers, weights)
                ax.plot([w, w], [0, y_max], linestyle='--', label=name)
        else:
            for name in mw_averages:
                w = averages[name](numbers, weights)
                ax.plot([w, w], [0, y_max], linestyle='--', label=name)

    ax.set_xlabel('Molecular Weight')
    fig.legend()

    return fig, ax


def gaussian_wf_vs_nf(μ, σ):
    """
    Plot a Gaussian distribution of weight factor vs number factor.

    :param μ: mean
    :param σ: standard deviation
    :return: fig, ax
    """
    return general_gaussian_wf_vs_nf(μ, σ, p=1)


def general_gaussian_wf_vs_nf(μ, σ, p):
    """
    Plot a general Gaussian distribution of weight factor vs number factor.

    :param μ: mean
    :param σ: standard deviation
    :param p: shape
    :return: fig, ax
    """
    number_of_points = 100
    numbers = general_gaussian(number_of_points, p, number_of_points*σ/(2*μ))
    weights = np.linspace(0, 2*μ, number_of_points)
    return graph_wf_vs_nf(numbers, weights)


def flory_schulz_wf_vs_nf(a, k_max, mw_averages=True):
    """
    Plot a Flory-Schulz distribution of weight factor vs number factor.

    :param a: parameter (0<a<1)
    :return: fig, ax
    """
    weights = np.arange(k_max)
    numbers = flory_schulz(a, weights)
    return graph_wf_vs_nf(numbers, weights, mw_averages)


if __name__ == "__main__":
    # Flory-Schulz
    fig, ax = flory_schulz_wf_vs_nf(0.05, 100, ['Mp', 'Mn', 'Mw', 'Mz'])
    ax.set_yticks(np.linspace(0, 0.02, 5))

    fig.savefig('mw.svg')
