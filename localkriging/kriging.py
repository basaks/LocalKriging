# -*- coding: utf-8 -*-

"""Main module."""
from os.path import basename, splitext
from collections import OrderedDict
import numpy as np
from scipy.spatial import cKDTree
import rasterio as rio
from rasterio.windows import Window
from geopandas import read_file
from pykrige import OrdinaryKriging, UniversalKriging
from configs.config import shapefile, covariates, regression_model, \
    target, num_points, kriging_method
from localkriging import mpiops


krige_methods = {'ordinary': OrdinaryKriging,
                 'universal': UniversalKriging}

targets = read_file(shapefile)
kriging = krige_methods[kriging_method]

xy = [(p.x, p.y) for p in targets['geometry']]
x = [p.x for p in targets['geometry']]
y = [p.y for p in targets['geometry']]
chem = targets[target]

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
residuals = chem - regression.predict(X)
tree = cKDTree(xy)


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
            pred = regression.predict(X).reshape(step, ds.width)  # regression

            for rr in range(step):
                print(rr, r + rr)
                for cc in range(ds.width):
                    lat, lon = ds.xy(r + rr, cc)
                    # print(lat, lon)
                    d, ii = tree.query([lat, lon], num_points)
                    # points = [xy[i] for i in ii]
                    xs = [x[i] for i in ii]
                    ys = [y[i] for i in ii]
                    zs = [residuals[i] for i in ii]  # residuals

                    krige_class = kriging(xs, ys, zs)
                    res, res_std = krige_class.execute('points', [lat], [lon])
                    pred[rr, cc] += res  # local kriged residual correction


            dst.write(pred.astype(rio.float32),
                      window=Window(0, r, ds.width, step),
                      indexes=1)
            print('wrote {} rows'.format(step))


predict(covariates)
