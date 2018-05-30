import logging
import numpy as np
import pickle

from gwr.gwr import GWR
from gwr.sel_bw import Sel_BW
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
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
        self.tree = None
        self.xy_dict = {}
        # self.max_distance = self.max_distance()

    def max_distance(self):
        D = squareform(pdist(self.xy))
        return np.nanmax(D)

    def fit(self, X, y, *args, **kwargs):
        self.regression.fit(X[:, 2:], y)
        reg_pred = self.regression.predict(X)
        residual = y - reg_pred
        self.tree = cKDTree(X[:, :2])
        self.residual = {k: v for k, v in enumerate(residual)}
        self.xy_dict = {k: v for k, v in enumerate(X[:, :2])}
        self.trained = True
        # pickle.dump(self.regression, open('regression_model.pk', 'wb'))
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
        # self.regression = pickle.load(open('regression_model.pk', 'rb'))
        reg_pred = self.regression.predict(X)
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
                        y_pred=self.predict(X)[0],
                        sample_weight=sample_weight)


class GWRMod:

    def __init__(self, coords,
                family,
                fixed,
                kernel):

        self.coords = coords
        self.fixed = fixed
        self.family = family
        self.kernel = kernel
        self.gwr = None

    def fit(self, X, y):

        y = y.values.reshape(-1, 1)

        if isinstance(X, np.ma.MaskedArray):
            X = np.ma.getdata(X)

        if self.gwr is None:  # not trained
            bw = Sel_BW(self.coords,
                        y,
                        X,
                        kernel='bisquare', fixed=False)
            bw = bw.search(search='golden_section', criterion='AICc')
            self.gwr = GWR(y=y,
                           X=X,
                           coords=self.coords,
                           bw=bw,
                           family=self.family,
                           fixed=self.fixed,
                           kernel=self.kernel)
        self.gwr.fit()
        log.info('Trained GWR model')

    def predict(self, X):
        if self.gwr is None:  # need to train
            raise AssertionError("Model not trained. Train first")
        else:
            return self.gwr.predict(points=X[:, :2],
                                    P=X[:, 2:]).predictions.flatten()
