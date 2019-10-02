def flory_schulz(a, k):
    """
    PMF for the Flory-Schulz distribution.
    """
    return a**2 * k * (1 - a)**(k - 1)


if __name__ == "__main__":
    print(flory_schulz(0.5, 1))
    print(flory_schulz(0.5, 10))
    print(flory_schulz(0.95, 1))
    print(flory_schulz(0.95, 10))
