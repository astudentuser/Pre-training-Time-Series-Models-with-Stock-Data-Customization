import numpy as np
import matplotlib.pyplot as plt


def wiener_process(delta, sigma, time, paths):
    """Returns a Wiener process
    Parameters
    delta : float. The increment to downsample sigma
    sigma : float. Percentage volatility
    time : int. Number of samples to create
    paths : int. Number of price simulations to create
    Returns
    wiener_process : np.ndarray
    """
    return sigma * np.random.normal(loc=0, scale=np.sqrt(delta), size=(time, paths))


def gbm_returns(delta, sigma, time, mu, paths):
    """Returns from a Geometric brownian motion
    Parameters
    delta : float. The increment to downsample sigma
    sigma : float. Percentage volatility
    time : int. Number of samples to create
    mu : float. Percentage drift
    paths : int. Number of price simulations to create
    Returns
    gbm_returns : np.ndarray
    """
    process = wiener_process(delta, sigma, time, paths)
    return np.exp(process + (mu - sigma ** 2 / 2) * delta)


def gbm_levels(s0, delta, sigma, time, mu, paths):
    """Returns price paths starting at s0
    Parameters
    s0 : float. The starting stock price
    delta : float. The increment to downsample sigma
    sigma : float. Percentage volatility
    time : int. Number of samples to create
    mu : float. Percentage drift
    paths : int. Number of price simulations to create
    Returns
    gbm_levels : np.ndarray
    """
    returns = gbm_returns(delta, sigma, time, mu, paths)
    stacked = np.vstack([np.ones(paths), returns])
    return s0 * stacked.cumprod(axis=0)

if __name__ == '__main__':
    # setup params for brownian motion
    s0 = 131.00
    sigma = 0.2
    mu = 0.2
    # setup the simulation
    paths = 1
    delta = 1.0 / 252.0
    time = 252 * 5
    price_paths = gbm_levels(s0, delta, sigma, time, mu, paths)
    plt.plot(price_paths, linewidth=0.25)
    plt.show()