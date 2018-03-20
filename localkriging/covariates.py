from os.path import basename, splitext
from collections import OrderedDict
import numpy as np
import rasterio as rio
from localkriging import mpiops


def _join_dicts(dicts):
    if dicts is None:
        return
    d = {k: v for D in dicts for k, v in D.items()}
    return OrderedDict(sorted(d.items()))


def gather_covariates(xy, covariates):
    """
    Gather covariates using MPI.

    Parameters
    ----------
    xy: list
        list of (x, y) points for which  covariate values are required
    covariates: list
        list of covariates to be intersected

    Returns
    -------
    features: dict
        dict of features names and interested values as numpy array

    """

    p_covaraites = mpiops.array_split(covariates, mpiops.rank)
    p_features = _process_gather_covariates(xy, p_covaraites)
    features = _join_dicts(mpiops.comm.allgather(p_features))
    return features


def _process_gather_covariates(xy, covariates):
    """
    Parameters
    ----------
    xy: list
        list of (x, y) points for which  covariate values are required
    covariates: list
        list of covariates to be intersected by this process

    Returns
    -------
    features: dict
        dict of features names and interested values as numpy array

    """
    # TODO: break this up in partitions for very large rasters
    features = {}
    for c in covariates:
        src = rio.open(c)
        features[splitext(basename(c))[0]] = np.array(list(src.sample(xy)))
        src.close()
    return features



