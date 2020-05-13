import jax.numpy as jnp
from jax.scipy.stats import norm
from utils import pdf_cdf_ratio_grad, pdf_cdf_ratio


def likelihood(x, n, p):

    return jnp.sum(p * norm.logcdf(x) + (n - p) * norm.logcdf(-x))


def likelihood_grad(x, n, p):

    return p * pdf_cdf_ratio(x) - (n - p) * pdf_cdf_ratio(-x)


def likelihood_diag_hess(x, n, p):

    return (p * pdf_cdf_ratio_grad(x) +
            (n - p) * pdf_cdf_ratio_grad(-x))
