# -*- coding: utf-8 -*-

"""Console script for localkriging."""
import csv
import importlib.util
import logging
import sys
from collections import OrderedDict
from os.path import basename, splitext

import click
import numpy as np
import rasterio as rio
from geopandas import read_file
from gwr.gwr import GWR
from pykrige import OrdinaryKriging, UniversalKriging, RegressionKriging
from pysal.contrib.glm.family import Gaussian
from rasterio.windows import Window

from localkriging import lklog
from localkriging import mpiops
from localkriging.covariates import gather_covariates
from localkriging.model import LocalRegressionKriging, GWRMod
from localkriging.writer import RasterWriter

DEFAULT_NODATA = -99999
log = logging.getLogger(__name__)

krige_methods = {'ordinary': OrdinaryKriging,
                 'universal': UniversalKriging,
                 'regression': RegressionKriging,
                 'gwr': GWR}


def load_config(config_file):
    module_name = 'config'
    spec = importlib.util.spec_from_file_location(
        module_name, config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


@click.command()
@click.option('-v', '--verbosity',
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              default='INFO', help='Level of logging')
@click.argument('config_file')
@click.argument('output_file')
@click.option('-k', '--kriged_residuals', type=str,
              default='kriged_residuals.tif',
              help='path to kriged residuals geotif')
@click.option('-p', '--partitions', type=int, default=1,
              help='Number of partitions used in prediction. A higher value '
                   'requires less memory.')
def main(config_file, output_file, kriged_residuals, partitions, verbosity):
    """Commandline options and logging setup"""
    lklog.configure(verbosity)

    log.info('Will use partitions={} during prediction. Use more '
             'partitions if limited memory is available.'.format(partitions))

    config = load_config(config_file)
    targets_all = mpiops.run_once(read_file, config.shapefile)

    xy = np.array([(p.x, p.y) for p in targets_all['geometry']])
    targets = targets_all[config.target]

    # intersect covariates
    features = gather_covariates(xy, config.covariates)

    # stack for learning
    X = np.ma.hstack([xy] + [v for v in features.values()])

    # ignore all rows with missing data
    # TODO: remove when we have imputation working
    valid_data_rows = X.mask.sum(axis=1) == 0

    if mpiops.rank == 0:
        if config.regression_model == 'gwr':

            regression_model = GWRMod(
                coords=xy[valid_data_rows],
                family=Gaussian(),
                fixed=False,
                kernel='bisquare')
        else:
            regression_model = config.model_maps[config.regression_model]()

        model = LocalRegressionKriging(
            xy[valid_data_rows].data,
            regression=regression_model,
            kriging_model=krige_methods[config.kriging_method],
            num_points=config.num_points,
            **config.kriging_params
        )

        # if config.cross_val:
        #     # TODO: write x-val score to a file
        #     log.info('Cross validation r2 score: {}'.format(np.mean(
        #         cross_val_score(model,
        #                         X[valid_data_rows].data,
        #                         y=targets[valid_data_rows],
        #                         cv=config.cross_val_folds))))

        model.fit(X=X[valid_data_rows].data, y=targets[
            valid_data_rows])
        # pickle.dump(model,
        #     open('local_kriged_regression_{}.model'.format(
        #         config.regression_model), 'wb'))
        # _output_residuals_and_predictions(model, X[valid_data_rows],
        #     targets_all[[config.target, 'geometry']])


    mpiops.comm.barrier()
    # choose a representative dataset
    ds = rio.open(config.covariates[0])

    # rasterio profile object
    profile = ds.profile
    # assume 1 band rasters
    profile.update(dtype=rio.float32, count=1, compress='lzw',
                   nodata=DEFAULT_NODATA)

    # mpi compatible writer class instance
    writer = RasterWriter(output_tif=output_file,
                          kriged_residuals=kriged_residuals,
                          profile=profile)

    # predict and write output geotif
    predict(ds, config, writer, model, partitions)

    log.info('Finished prediction')


def _output_residuals_and_predictions(model, X, gdf):
    """
    wirte out residuals and predictions at the target locations
    """
    with open('target_results.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        regresstion_pred = model.regression.predict(X)
        pred, res = model.predict(X)
        csvwriter.writerow(['lon', 'lat', 'residual', 'lrk_pred',
                            'reg_pred'])
        for xy, r, p, reg in zip(model.xy, res, pred, regresstion_pred):
            csvwriter.writerow([xy[0], xy[1], r, p, reg])
    log.info('Wrote residuals and predictions at target')

    gdf['residual'] = res
    gdf['lrk_prediction'] = pred
    gdf['regression_pred'] = regresstion_pred
    gdf.to_file('output_shapefile.shp')
    log.info('Wrote residuals and predictions at target in shapefile')


def predict(ds, config, writer, model, partitions=10):
    feats = {}
    covariates = config.covariates
    process_rows = mpiops.array_split(range(ds.height))

    # model = pickle.load(open('local_kriged_regression_{}.model'.format(
    #     config.regression_model), 'rb'))

    # idea: instead of rows, using tiles may be more efficient due to
    # variogram computation in the LocalRegressionKriging class
    for p, r in enumerate(np.array_split(process_rows, partitions)):
        log.info('Processing partition {}'.format(p+1))
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

        log.info('predicting rows {r0} through {rl} '
                 'using process {rank}'.format(r0=r[0], rl=r[-1],
                                               rank=mpiops.rank))

        # vectors of rows and cols we need lats and lons for
        rs = np.repeat(r, ds.width)
        cs = np.repeat(np.atleast_2d(np.array(range(ds.width))),
                       step, axis=0).flatten()
        lats, lons = ds.xy(rs, cs)
        # stack with lats and lons
        X = np.ma.hstack([np.atleast_2d(lats).T, np.atleast_2d(lons).T, X])

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
