from sklearn.ensemble import RandomForestRegressor

shapefile = '/home/sudipta/Documents/GA-cover2/geochem_sites.shp'
target = 'K_ppm_imp_'

covariates = [
    '/home/sudipta/Documents/GA-cover2/LATITUDE_GRID1.tif',
    '/home/sudipta/Documents/GA-cover2/LONGITUDE_GRID1.tif',
    '/home/sudipta/Documents/GA-cover2/Clim_Prescott_LindaGregory.tif',
    '/home/sudipta/Documents/GA-cover2/dem_foc2.tif',
    # '/home/sudipta/Documents/GA-cover2/gg_clip.tif',
    # '/home/sudipta/Documents/GA-cover2/k_15v5.tif',
    # '/home/sudipta/Documents/GA-cover2/modis10_te.tif',
    # '/home/sudipta/Documents/GA-cover2/modis11_te.tif',
    # '/home/sudipta/Documents/GA-cover2/mrvbf_9.tif',
    # '/home/sudipta/Documents/GA-cover2/MvrtpLL.tif',
    # '/home/sudipta/Documents/GA-cover2/outcrop_dis2.tif',
]


lat = '/home/sudipta/Documents/GA-cover2/LATITUDE_GRID1.tif'
lon = '/home/sudipta/Documents/GA-cover2/LONGITUDE_GRID1.tif'

regression_model = RandomForestRegressor()


# kriging parameters

# number of points used in local kriging
num_points = 5

# should be ordinary or universal
kriging_method = 'ordinary'


# some checks
def _check_kriging_method():
    if kriging_method not in ['ordinary', 'universal']:
        raise ValueError('kriging method must be ordinary or universal')


def _check_covariates_not_repeated():
    pass

_check_kriging_method()
_check_covariates_not_repeated()

