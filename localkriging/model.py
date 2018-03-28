import logging
import numpy as np
from scipy.spatial import cKDTree
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.metrics import r2_score
log = logging.getLogger(__name__)


class LocalRegressionKriging(RegressorMixin, BaseEstimator):
    def __init__(self,
                 xy,
                 regression,
                 kriging_model,
                 num_points,
                 **kwargs):
        """
        Parameters
        ----------
        xy: np.ndarray
            array of dim [n, 2] of (x, y) points for which  covariate values
            available (observation coordinates)
        regression_model: sklearn compatible regression class
        kriging_model: pykrige kriging class instance
            should be 'ordinary' or 'universal'
        variogram_model: str
            pykrige compatible variogram model
        num_points: int
            number of points for the local kriging
        """
        # self.xy = {k: v for k, v in enumerate(xy)}
        self.xy = xy
        self.regression = regression
        self.kriging_model = kriging_model
        self.kwargs = kwargs
        self.num_points = num_points
        self.trained = False
        self.residual = {}
        self.tree = cKDTree(xy)
        self.xy_dict = {k: v for k, v in enumerate(xy)}

    def fit(self, X, y, *args, **kwargs):
        self.regression.fit(X[:, 2:], y)
        residual = y - self.regression.predict(X[:, 2:])
        self.residual = {k: v for k, v in enumerate(residual)}
        self.trained = True
        log.info('local regression kriging model trained')

    def predict(self, X, *args, **kwargs):
        """
        Parameters
        ----------
        X: np.array
            features of the regression model
            numpy array of dim(n, 2+ nfeatures), +2 becuase of lat and lon
        """
        if not self.trained:
            raise Exception('Not trained. Train first')

        reg_pred = self.regression.predict(X[:, 2:])
        # return reg_pred, np.empty_like(reg_pred, dtype=np.float32)
        # TODO: return std for regression models that support std
        res = self._krige_locally_batch(X[:, 0], X[:, 1])
        return (reg_pred + res).astype(np.float32), res

    def _krige_locally_batch(self, lats, lons):
        res = np.empty_like(lats, dtype=np.float32)

        # just create dummy initial set
        last_set = set(range(self.num_points))
        krige = None

        for i, (lat, lon) in enumerate(zip(lats, lons)):
            resid, last_set, krige = self._krige_locally(
                lat, lon, last_set, krige)
            res[i] = resid
            if not i % 10000:
                log.info('processed {} pixels'.format(i))

        return res

    def _krige_locally(self, lat, lon, last_set, krige):
        """
        This is the local residual kriging step.

        :param lat:
        :param lon:
        :param reg_pred:
        :return:
        """
        d, ii = self.tree.query([lat, lon], self.num_points)

        # create a set of points with the closest points index
        points = set(ii)

        # only compute kriging model when previous points set does not match
        # making the computation potentially 10x more efficient
        if points != last_set:
            xs = [self.xy_dict[i][0] for i in ii]
            ys = [self.xy_dict[i][1] for i in ii]
            zs = [self.residual[i] for i in ii]
            krige = self.kriging_model(xs, ys, zs, **self.kwargs)
            last_set = points
        res, res_std = krige.execute('points', [lat], [lon])
        return res, last_set, krige  # local kriged residual correction

    def score(self, X, y, sample_weight=None):
        return r2_score(y_true=y,
                        y_pred=self.predict(X),
                        sample_weight=sample_weight)
