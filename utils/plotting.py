import seaborn as sns
import matplotlib.pyplot as plt


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
