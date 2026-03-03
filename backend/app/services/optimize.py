import numpy as np
import pandas as pd
from scipy.optimize import minimize


def min_variance_weights(returns):

    cov = returns.cov()

    n = len(cov)

    def portfolio_var(w):

        return w.T @ cov @ w

    x0 = np.ones(n)/n

    constraints = ({
        'type':'eq',
        'fun': lambda w: np.sum(w)-1
    })

    bounds = [(0,1)]*n

    result = minimize(
        portfolio_var,
        x0,
        bounds=bounds,
        constraints=constraints
    )

    return result.x
