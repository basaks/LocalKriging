from os.path import basename, splitext
from collections import OrderedDict
import numpy as np
import rasterio as rio
from rasterio.windows import Window
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
        features[splitext(basename(c))[0]] = np.ma.array(
            list(sample_gen(src, xy)))
        src.close()
    return features


def sample_gen(dataset, xy, indexes=None):
    """"
    Inspired from
    https://mapbox.github.io/rasterio/_modules/rasterio/sample.html#sample_gen
    Generator for sampled pixels"""
    index = dataset.index
    read = dataset.read

    if isinstance(indexes, int):
        indexes = [indexes]

    for x, y in xy:
        row_off, col_off = index(x, y)
        window = Window(col_off, row_off, 1, 1)
        data = read(indexes, window=window, masked=True, boundless=True)
        yield data[:, 0, 0]
