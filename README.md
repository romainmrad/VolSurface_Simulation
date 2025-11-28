# Implied Volatility Surface Forecasting Project

## Overview

This project builds a full pipeline to model, simulate, and forecast implied volatility surfaces using a low dimensional parametric model driven by latent beta factors. The goal is to extract daily surface dynamics from market data, model their statistical behavior, and produce forward looking implied volatility surfaces from simulated factor paths.

The workflow is split into three main notebooks that handle data preparation, beta extraction, and stochastic simulation.

## Project Structure

```
01_DataLoader.ipynb
02_BetaExtraction.ipynb
03_BetaSimulation.ipynb
```

## Requirements

```
numpy
pandas
scipy
matplotlib
plotly
hmmlearn
wrds
```

## Mathematical Framework

### Parametric Implied Volatility Surface

The implied volatility surface is modeled as a parametric function of moneyness \( M \) and time to maturity \( T \):

$$
\sigma(M, T) = \beta_1 + \beta_2 \exp\left( - \sqrt{ \frac{T}{T_{\text{conv}}} } \right) +
\beta_3 \, \phi(M) +  \beta_4 \left(1 - \exp(-M^2)\right) \log\left( \frac{T}{T_{\text{max}}} \right) + b_5 \, \psi(M) \log\left( \frac{T}{T_{\text{max}}} \right)
$$

where the nonlinear slope functions are defined as

$$
\phi(M) =
\begin{cases}
M, & M \ge 0 \\\\
\frac{\exp(2M) - 1}{\exp(2M) + 1}, & M < 0
\end{cases}
$$

and

$$
\psi(M) =
\begin{cases}
1 - \exp\left((3M)^3\right), & M < 0 \\\\
0, & M \ge 0
\end{cases}
$$

The coefficient vector is

$$
\beta = (\beta_1, \beta_2, \beta_3, \beta_4, \beta_5)^\top
$$

Each component controls a structural shape of the surface such as level, term structure, skewness, curvature, and left tail asymmetry.

---

### Regime Switching with Gaussian Hidden Markov Model

The latent market state $S_t$ follows a first order Markov chain with two regimes:

$$
P(S_t = j \mid S_{t-1} = i) = p_{ij}
$$

The transition matrix is

$$
P =
\begin{pmatrix}
p_{00} & p_{01} \\\\
p_{10} & p_{11}
\end{pmatrix}
$$

Conditional on the regime, the innovation process follows a multivariate normal distribution:

$$
\varepsilon_t \mid (S_t = k) \sim \mathcal{N}(0, \Sigma_k)
$$

---

### Dynamics of Beta Increments

The beta factors evolve through a regime dependent random walk:

$$
\Delta \beta_t = \beta_t - \beta_{t-1} = \mu+ \varepsilon_t
$$

with

$$
\varepsilon_t \mid (S_t = k) \sim \mathcal{N}(0, \Sigma_k)
$$

and therefore

$$
\beta_t = \beta_{t-1} + \mu + \varepsilon_t
$$

This structure allows the surface to evolve smoothly while adapting its volatility according to latent market regimes.

---

### Forecasted Surface

Given a simulated future beta vector $\beta_{t+h}$, the predicted surface is reconstructed as:

$$
\hat{\sigma}_{t+h}(M, T) = \sigma(M, T; \beta_{t+h})
$$
