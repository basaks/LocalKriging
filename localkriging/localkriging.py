# -*- coding: utf-8 -*-

"""Main module."""
import numpy as np
import rasterio as rio
from sklearn.ensemble import RandomForestRegressor
from geopandas import read_file
from configs.config import shapefile, covariates

targets = read_file(shapefile)

lat_f = covariates[0]
lon_f = covariates[1]

xy = [(p.x, p.y) for p in targets['geometry']]
src = rio.open(lat_f)
lats = np.array(list(src.sample(xy)))
src = rio.open(lon_f)
lons = np.array(list(src.sample(xy)))

rfr = RandomForestRegressor()
X = np.hstack([lons, lats])
rfr.fit(X=X, y=targets['K_ppm_imp_'])
print(rfr.score(X=X, y=targets['K_ppm_imp_']))
