# -*- coding: utf-8 -*-

"""Console script for localkriging."""
import sys
from os.path import basename, splitext
from collections import OrderedDict
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
def main(config_file, output_file):
    config = load_config(config_file)
    targets_all = read_file(config.shapefile)

    xy = [(p.x, p.y) for p in targets_all['geometry']]
    targets = targets_all[config.target]

    model = LocalRegressionKriging(
        xy,
        regression_model=config.regression_model,
        kriging_model=config.kriging_method,
        variogram_model=config.variogram_model,
        num_points=config.num_points
    )

    # interset covariates
    features = gather_covariates(xy, config.covariates)

    # stack for learning
    X = np.hstack([v for v in features.values()])

    model.fit(X, y=targets)

    # choose a representative dataset
    ds = rio.open(config.covariates[0])

    # rasterio profile object
    profile = ds.profile
    # assume 1 band rasters
    profile.update(dtype=rio.float32, count=1, compress='lzw')

    # mpi compatible writer class instance
    writer = RasterWriter(output_tif=output_file,
                          profile=profile)

    # predict and write output geotif
    predict(ds, config, writer, model)

    return 0


def predict(ds, config, writer, model, step=10):
    feats = {}
    covariates = config.covariates
    process_rows = mpiops.array_split(range(ds.height))

    dummy_rows = 0
    max_process_rows = ds.height // mpiops.size + 1
    if ds.height % mpiops.size:
        dummy_rows = max_process_rows - len(process_rows)

    for r in process_rows:
        for c in covariates:
            with rio.open(c) as src:
                # Window(col_off, row_off, width, height)
                # assume band one for now
                w = src.read(1, window=Window(0, r, src.width, 1))
                feats[splitext(basename(c))[0]] = w.flatten()

        feats = OrderedDict(sorted(feats.items()))
        X = np.vstack([v for v in feats.values()]).T
        print('processed row {} using process {}'.format(r, mpiops.rank))
        pred = np.zeros(shape=(1, ds.width))

        # this is the local residual kriging step
        for cc in range(ds.width):
            lat, lon = ds.xy(r, cc)
            pred[0, cc] = model.predict(np.atleast_2d(X[cc, :]),
                                        lat, lon)

        writer.write({'data': pred.astype(rio.float32),
                      'window': (0, r, ds.width, 1)})

    for _ in range(dummy_rows):
        writer.write({'data': None,
                      'window': None})


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
