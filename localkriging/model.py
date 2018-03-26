import numpy as np
from scipy.spatial import cKDTree
from pykrige import OrdinaryKriging, UniversalKriging
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.metrics import r2_score

krige_methods = {'ordinary': OrdinaryKriging,
                 'universal': UniversalKriging}


class LocalRegressionKriging(RegressorMixin, BaseEstimator):
    def __init__(self,
                 xy,
                 regression_model,
                 kriging_model,
                 variogram_model,
                 num_points):
        """
        Parameters
        ----------
        xy: list
            list of (x, y) points for which  covariate values are required
        regression_model: sklearn compatible regression class
        kriging_model: str
            should be 'ordinary' or 'universal'
        variogram_model: str
            pykrige compatible variogram model
        num_points: int
            number of points for the local kriging
        """
        self.xy = {k: v for k, v in enumerate(xy)}
        self.regression = regression_model
        self.kriging_model = krige_methods[kriging_model]
        self.variogram_model = variogram_model
        self.num_points = num_points
        self.trained = False
        self.residual = {}
        self.tree = cKDTree(xy)

    def fit(self, X, y, *args, **kwargs):
        self.regression.fit(X, y)
        residual = y - self.regression.predict(X)
        self.residual = {k: v for k, v in enumerate(residual)}

        self.trained = True

    def predict(self, X, lats, lons, *args, **kwargs):
        """
        Parameters
        ----------
        X: np.array
            features of the regression model, 2d numpy array
        lats: np.array
            latitude 1d np.array
        lons: np.array
            longitude 1d np.array

        """
        if not self.trained:
            raise Exception('Not trained. Train first')

        # self._input_sanity_check(X, lat, lon)

        reg_pred = self.regression.predict(X)
        # return reg_pred, np.empty_like(reg_pred, dtype=np.float32)
        # TODO: return std for regression models that support std

        return self._krige_locally_batch(lats, lons, reg_pred)

    def _krige_locally_batch(self, lats, lons, reg_pred):
        kriged_pred = np.empty_like(lats, dtype=np.float32)
        res = np.empty_like(lats, dtype=np.float32)
        for i, (lat, lon, pred) in enumerate(zip(lats, lons, reg_pred)):
            kriged_pred[i], res[i] = self._krige_locally(lat, lon, pred)
        return kriged_pred, res

    def _krige_locally(self, lat, lon, reg_pred):
        """
        This is the local residual kriging step.

        :param lat:
        :param lon:
        :param reg_pred:
        :return:
        """
        d, ii = self.tree.query([lat, lon], self.num_points)
        xs = [self.xy[i][0] for i in ii]
        ys = [self.xy[i][1] for i in ii]
        zs = [self.residual[i] for i in ii]
        krige_class = self.kriging_model(xs, ys, zs, self.variogram_model)
        res, res_std = krige_class.execute('points', [lat], [lon])
        return reg_pred + res, res  # local kriged residual correction

    def score(self, X, y, lats, lons, sample_weight=None):
        return r2_score(y_true=y,
                        y_pred=self.predict(X, lats, lons),
                        sample_weight=sample_weight)

    def _input_sanity_check(self, X, lat, lon):
        if X.shape[0] != len(lat):
            raise ValueError('X and lat must of same length')

        if X.shape[0] != len(lon):
            raise ValueError('X and lat must of same length')
