import numpy as np

def historical_var(returns, alpha=0.95):

    returns = np.array(returns)

    var = np.percentile(returns, (1-alpha)*100)

    return -var


def cvar(returns, alpha=0.95):

    returns = np.array(returns)

    var = np.percentile(returns, (1-alpha)*100)

    losses = returns[returns <= var]

    return -losses.mean()
