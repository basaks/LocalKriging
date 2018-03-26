# -*- coding: utf-8 -*-

"""Console script for localkriging."""
import sys
from os.path import basename, splitext
from collections import OrderedDict
import pickle
import numpy as np
import click
import importlib.util
from geopandas import read_file
import rasterio as rio
from rasterio.windows import Window
from localkriging import mpiops
from localkriging.model import LocalRegressionKriging
from localkriging.covariates import gather_covariates
from localkriging.writer import RasterWriter

DEFAULT_NODATA = 1.0e-20


def load_config(config_file):
    module_name = 'config'
    spec = importlib.util.spec_from_file_location(
        module_name, config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


@click.command()
@click.argument('config_file')
@click.argument('output_file')
@click.option('-k', '--kriged_residuals', type=str,
              default='kriged_residuals.tif',
              help='path to kriged residuals geotif')
@click.option('-p', '--partitions', type=int, default=1,
              help='Number of partitions used in prediction. A higher value '
                   'requires less memory.')
def main(config_file, output_file, kriged_residuals, partitions):

    print('Will use partitions={} during prediction. Use more '
          'partitions if limited memory is available.'.format(partitions))

    config = load_config(config_file)
    targets_all = mpiops.run_once(read_file, config.shapefile)

    xy = np.array([(p.x, p.y) for p in targets_all['geometry']])
    targets = targets_all[config.target]

    # intersect covariates
    features = gather_covariates(xy, config.covariates)

    # stack for learning
    X = np.ma.hstack([v for v in features.values()])

    # ignore all rows with missing data
    # TODO: remove when we have imputation working
    valid_data_rows = X.mask.sum(axis=1) == 0

    if mpiops.rank == 0:
        model = LocalRegressionKriging(
            xy[valid_data_rows],
            regression_model=config.regression_model,
            kriging_model=config.kriging_method,
            variogram_model=config.variogram_model,
            num_points=config.num_points
        )
        model.fit(X[valid_data_rows], y=targets[valid_data_rows])
        pickle.dump(model, open('local_kriged_regression.model', 'wb'))
    mpiops.comm.barrier()
    # choose a representative dataset
    ds = rio.open(config.covariates[0])

    # rasterio profile object
    profile = ds.profile
    # assume 1 band rasters
    profile.update(dtype=rio.float32, count=1, compress='lzw', nodata=1.0e-20)

    # mpi compatible writer class instance
    writer = RasterWriter(output_tif=output_file,
                          kriged_residuals=kriged_residuals,
                          profile=profile)

    # predict and write output geotif
    predict(ds, config, writer, partitions)

    return 0


def predict(ds, config, writer, partitions=10):
    feats = {}
    covariates = config.covariates
    process_rows = mpiops.array_split(range(ds.height))

    model = pickle.load(open('local_kriged_regression.model', 'rb'))

    # TODO: compute in `step`s for faster (at least) regression prediction
    for p, r in enumerate(np.array_split(process_rows, partitions)):
        print('Processing partition {}'.format(p))
        step = len(r)
        for c in covariates:
            with rio.open(c) as src:
                # Window(col_off, row_off, width, height)
                # assume band one for now
                w = src.read(1, window=Window(0, r[0], src.width, step))
                feats[splitext(basename(c))[0]] = w.flatten()

        feats = OrderedDict(sorted(feats.items()))

        # stack for prediction
        X = np.ma.vstack([v for v in feats.values()]).T

        print('predicting rows {r0} through {rl} using process {rank}'.format(
            r0=r[0], rl=r[-1], rank=mpiops.rank))

        # vectors of rows and cols we need lats and lons for
        rs = np.repeat(r, ds.width)
        cs = np.repeat(np.atleast_2d(np.array(range(ds.width))),
                       step, axis=0).flatten()
        lats, lons = ds.xy(rs, cs)
        # TODO: remove this when we have imputation working
        # just assign nodata when there is nodata in any covariate
        no_data_mask = X.mask.sum(axis=1) != 0
        pred, res = model.predict(X, lats, lons)
        pred[no_data_mask] = DEFAULT_NODATA
        res[no_data_mask] = DEFAULT_NODATA

        writer.write({'data': pred.reshape((step, -1)).astype(rio.float32),
                      'residuals': res.reshape((step, -1)).astype(rio.float32),
                      'window': (0, r[0], ds.width, step)})


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
