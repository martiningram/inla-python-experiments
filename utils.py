import numpy as np
from scipy.stats import norm


def pdf_cdf_ratio(x):

    return np.exp(norm.logpdf(x) - norm.logcdf(x))


def pdf_cdf_ratio_grad(x):

    g = pdf_cdf_ratio(x)

    return -g * (x + g)
