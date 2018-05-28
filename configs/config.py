import os
import logging
from sklearn.ensemble import RandomForestRegressor
from gwr.gwr import GWR
from gwr.sel_bw import Sel_BW
from pysal.contrib.glm.family import Gaussian
from sklearn.linear_model import LinearRegression
log = logging.getLogger(__name__)


shapefile = '/home/sudipta/Documents/GA-cover2/geochem_sites.shp'
target = 'K_ppm_imp_'

log.info('Using shapefile {}'.format(os.path.basename(shapefile)))
log.info('Using target {}'.format(target))

covarites_dir = '/home/sudipta/Documents/GA-cover2/cropped'

# TODO: support inputs based on a list in a file
covariates = [
    # 'LATITUDE_GRID1.tif',
    # 'LONGITUDE_GRID1.tif',
    'Clim_Prescott_LindaGregory.tif',
    'gg_clip.tif',
#    'modis10_te.tif',
#    'mrvbf_9.tif',
#    'outcrop_dis2.tif',
]

covariates = [os.path.join(covarites_dir, c) for c in covariates]


bw = Sel_BW(xy[valid_data_rows], targets[valid_data_rows].values.reshape(-1,1),
            X[valid_data_rows], kernel='bisquare', fixed=False)
bw = bw.search(search='golden_section', criterion='AICc')
regression_model = GWR

    # (xy[valid_data_rows],
    #         targets[valid_data_rows].values.reshape(-1, 1),
    #         X[valid_data_rows].data,
    #         bw,
    #         family=Gaussian(),
    #         fixed=False,
    #         kernel='bisquare')

# regression_model = RandomForestRegressor()


# kriging parameters

# number of points used in local kriging
num_points = 5

# should be ordinary or universal
kriging_method = 'ordinary'

# variogram options are (linear, power, gaussian, spherical, exponential,
# hole-effect)

kriging_params = {
    'variogram_model': 'gaussian',
    'variogram_parameters': {'nugget': 10, 'sill': 50, 'range': 0.2},
    'weight': True,
}

cross_val = True
cross_val_folds = 5


# some checks
def _check_kriging_method():
    if kriging_method not in ['ordinary', 'universal']:
        raise ValueError('kriging method must be ordinary or universal')


def _check_covariates_not_repeated():
    pass


if 'variogram_model' not in kriging_params:
    raise ValueError('Variogram model must be provided')

_check_kriging_method()
_check_covariates_not_repeated()
