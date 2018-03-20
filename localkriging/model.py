import numpy as np
from scipy.spatial import cKDTree
from sklearn.base import RegressorMixin, BaseEstimator
from pykrige import OrdinaryKriging, UniversalKriging

krige_methods = {'ordinary': OrdinaryKriging,
                 'universal': UniversalKriging}


class LocalRegressionKriging(RegressorMixin, BaseEstimator):
    def __init__(self,
                 xy,
                 regression_model,
                 kriging_model,
                 variogram_model,
                 num_points):
        self.xy = xy
        self.regression = regression_model
        self.kriging_model = kriging_model
        self.variogram_model = variogram_model
        self.num_points = num_points
        self.trained = False
        self.residual = np.zeros_like(self.xy)
        self.tree = cKDTree(self.xy)

    def fit(self, x, y, *args, **kwargs):
        self.regression.fit(x, y)
        self.residual = y - self.regression.predict(x)
        self.trained = True

    def predict(self, x, lat, lon, *args, **kwargs):
        """
        :param x:
        :param lat:
        :param lon:
        :return:
        """
        if not self.trained:
            raise Exception('Not trained. Train first')

        reg_pred = self.regression.predict(x)
        d, ii = self.tree.query([lat, lon], self.num_points)

        xs = [self.xy[i][0] for i in ii]
        ys = [self.xy[i][1] for i in ii]
        zs = [self.residual[i] for i in ii]
        krige_class = self.kriging_model(xs, ys, zs, self.variogram_model)
        res, res_std = krige_class.execute('points', [lat], [lon])
        reg_pred += res  # local kriged residual correction

        return reg_pred
