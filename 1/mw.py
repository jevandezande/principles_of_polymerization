import numpy as np


def weight_fractions(numbers, weights):
    """
    Weights fractions
    :param numbers: number of particles at a given weight
    :param weights: given weight

    >>> sum(weight_fractions([1, 2, 3, 2, 1], [1, 2, 3, 4, 5]))
    1.0
    >>> weight_fractions([1, 2, 3, 2, 1], [1, 2, 3, 4, 5])
    array([0.03703704, 0.14814815, 0.33333333, 0.2962963 , 0.18518519])
    """
    numbers, weights = np.asarray(numbers), np.asarray(weights)

    nxw = numbers*weights
    return nxw/sum(nxw)


def mn(numbers, weights):
    """
    Calculate the number average, Mn
    :param numbers: number of particles at a given weight
    :param weights: given weight

    Σ N_x M_x/Σ N_x

    >>> mn([1, 2, 3, 2, 1], [1, 2, 3, 4, 5])
    3.0
    """
    numbers, weights = np.asarray(numbers), np.asarray(weights)

    return sum(numbers*weights)/sum(numbers)


def mp(numbers, weights):
    """
    Calculate the peak molecular weight, Mp
    :param numbers: number of particles at a given weight
    :param weights: given weight

    >>> mp([1, 2, 3, 2, 1], [1, 2, 3, 4, 5])
    3
    """
    numbers, weights = np.asarray(numbers), np.asarray(weights)

    return weights[np.argmax(numbers)]


def mv(numbers, weights, α=0.5):
    """
    Calculate the viscosity average, Mv
    :param numbers: number of particles at a given weight
    :param weights: given weight
    :param α: viscosity parameter

    (Σ N_x M_x^{α + 1}/Σ N_x M_x)^{1/α}

    >>> mv([1, 2, 3, 2, 1], [1, 2, 3, 4, 5], 0.5)
    3.3510219709459164
    """
    numbers, weights = np.asarray(numbers), np.asarray(weights)

    return (sum(numbers*weights**(α+1))/sum(numbers*weights))**(1/α)


def mw(numbers, weights):
    """
    Calculate the weight average, Mw
    :param numbers: number particles at a given weight
    :param weights: given weight

    Σ N_x M_x^2/Σ N_x M_x

    >>> mw([1, 2, 3, 2, 1], [1, 2, 3, 4, 5])
    3.4444444444444446
    """
    return mz(numbers, weights, z=1)


def mz(numbers, weights, z=2):
    """
    Calculate the higher average, Mz
    :param numbers: number particles at a given weight
    :param weights: given weight
    :param z: average number

    Σ N_x M_x^{3 + i}/Σ N_x M_x^{2 + i}

    >>> mz([1, 2, 3, 2, 1], [1, 2, 3, 4, 5], 1)
    3.4444444444444446
    >>> mz([1, 2, 3, 2, 1], [1, 2, 3, 4, 5], 2)
    3.774193548387097
    >>> mz([1, 2, 3, 2, 1], [1, 2, 3, 4, 5], 3)
    4.0256410256410255
    """
    numbers, weights = np.asarray(numbers), np.asarray(weights)

    return sum(numbers * weights**(z+1))/sum(numbers * weights**z)


averages = {
    'Mn': mn,
    'Mp': mp,
    'Mv': mv,
    'Mw': mw,
    'Mz': mz,
}
