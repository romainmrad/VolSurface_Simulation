import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np


def plot_beta(df) -> None:
    """
    Plot evolution of beta timeseries
    :param df: the timeseries dataframe
    """
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(3, 1, figsize=(14, 8))
    # Panel A: beta1 (Level) and beta3 (Moneyness slope)
    sns.lineplot(x=df['date'], y=df['beta1'], ax=axes[0], label='$\\beta_1$', linewidth=0.8, color='blue')
    sns.lineplot(x=df['date'], y=df['beta3'], ax=axes[0], label='$\\beta_3$', linewidth=0.8, color='orange')
    axes[0].set_ylabel('Value', fontsize=11)
    axes[0].set_title('$\\beta_1$ (Level) and $\\beta_3$ (Moneyness slope)', fontsize=12)
    axes[0].legend(loc='upper right', fontsize=10)
    # Panel B: beta2 (Time-to-maturity slope)
    sns.lineplot(x=df['date'], y=df['beta2'], ax=axes[1], label='$\\beta_2$', linewidth=0.8, color='green')
    axes[1].set_ylabel('Value', fontsize=11)
    axes[1].set_title('$\\beta_2$ (Time-to-maturity slope)', fontsize=12)
    axes[1].legend(loc='upper right', fontsize=10)
    # Panel C: beta4 (Smile attenuation) and beta5 (Smirk)
    sns.lineplot(x=df['date'], y=df['beta4'], ax=axes[2], label='$\\beta_4$', linewidth=0.8, color='yellow')
    sns.lineplot(x=df['date'], y=df['beta5'], ax=axes[2], label='$\\beta_5$', linewidth=0.8, color='red')
    axes[2].set_ylabel('Value', fontsize=11)
    axes[2].set_title('$\\beta_4$ (Smile attenuation) and $\\beta_5$ (Smirk)', fontsize=12)
    axes[2].legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.show()


def sigma_model(M, T, beta_vec):
    """
    Volatility surface model
    :param M: Moneyness
    :param T: Maturity
    :param beta_vec: beta vector
    :return: Volatility
    """
    T_CONV = 0.25
    T_MAX = 5
    b1, b2, b3, b4, b5 = beta_vec
    term1 = b1
    term2 = b2 * np.exp(-np.sqrt(T / T_CONV))
    slope_pos = M
    slope_neg = (np.exp(2 * M) - 1) / (np.exp(2 * M) + 1)
    term3 = b3 * np.where(M >= 0, slope_pos, slope_neg)
    term4 = b4 * (1 - np.exp(-M ** 2)) * np.log(T / T_MAX)
    smirk = (1 - np.exp((3 * M) ** 3)) * np.log(T / T_MAX)
    term5 = b5 * np.where(M < 0, smirk, 0.0)
    return term1 + term2 + term3 + term4 + term5


def BS_PUT(F, K, T, r, sigma):
    """
    Compute Black-Scholes Put option price
    :param F: forward
    :param K: strike
    :param T: maturity
    :param r: risk-free rate
    :param sigma: volatility
    :return: Put option price
    """
    d1 = (np.log(F / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - F * norm.cdf(-d1)
