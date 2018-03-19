# -*- coding: utf-8 -*-

"""Main module."""
from os.path import basename, splitext
from collections import OrderedDict
import numpy as np
import rasterio as rio
from rasterio.windows import Window
from geopandas import read_file
from configs.config import shapefile, covariates, regression_model, target
from localkriging import mpiops

targets = read_file(shapefile)

xy = [(p.x, p.y) for p in targets['geometry']]

# rfr = RandomForestRegressor()
# X = np.hstack([lons, lats])
# rfr.fit(X=X, y=targets['K_ppm_imp_'])
# print(rfr.score(X=X, y=targets['K_ppm_imp_']))


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


features = gather_covariates(xy, covariates)

X = np.hstack([v for v in features.values()])

regression = regression_model.fit(X, y=targets[target])
residuals = regression.predict(X) - targets[target]


def predict(covariates, step=10):
    ds = rio.open(covariates[0])
    feats = {}
    profile = ds.profile

    # assume 1 band rasters
    profile.update(dtype=rio.float32, count=1, compress='lzw')

    with rio.open('out.tif', 'w', **profile) as dst:

        for r in range(0, ds.height, step):  # 10 rows at a time
            step = min(step, ds.height - r)
            print(r, step)
            for c in covariates:
                with rio.open(c) as src:
                    # Window(col_off, row_off, width, height)
                    # assume band one for now
                    w = src.read(1, window=Window(0, r, src.width, step))
                    feats[splitext(basename(c))[0]] = w.flatten()

            feats = OrderedDict(sorted(feats.items()))
            X = np.vstack([v for v in feats.values()]).T
            pred = regression.predict(X).reshape(step, ds.width)

            dst.write(pred.astype(rio.float32),
                      window=Window(0, r, ds.width, step),
                      indexes=1)
            print('wrote {} rows'.format(step))


# implement local kriging on residuals

predict(covariates)
