"""
Shared model code: feature builder + ensemble (Gaussian Process + quadratic ridge).
Inputs  : feed_rate, speed, material, process
Outputs : width, depth, reinforcement
(HIP and efficiency are NOT inputs: they are derived from feed rate and speed,
 and efficiency = 0.6 * HIP exactly, so they add no information.)
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

TARGETS = ["width", "depth", "reinforcement"]
MATERIALS = ["IN718", "SS304L"]
PROCESSES = ["ColdArc", "GMAW"]
GROUPS = [f"{m}_{p}" for m in MATERIALS for p in PROCESSES]


def build_features(feed_rate, speed, material, process):
    """Return an (n, 6) array: feed, speed, 4 one-hot material/process columns."""
    feed_rate = np.atleast_1d(feed_rate).astype(float)
    speed = np.atleast_1d(speed).astype(float)
    material = np.atleast_1d(material)
    process = np.atleast_1d(process)
    onehot = np.zeros((len(feed_rate), len(GROUPS)))
    for i, (m, p) in enumerate(zip(material, process)):
        key = f"{m}_{p}"
        if key not in GROUPS:
            raise ValueError(f"Unknown material/process '{key}'. Valid: {GROUPS}")
        onehot[i, GROUPS.index(key)] = 1.0
    return np.column_stack([feed_rate, speed, onehot])


def _make_gp(n_features):
    kernel = C(1.0) * RBF(np.ones(n_features), (0.1, 100)) + WhiteKernel(0.1)
    return GaussianProcessRegressor(kernel, normalize_y=True,
                                    n_restarts_optimizer=5, random_state=0)


def _make_poly():
    return make_pipeline(PolynomialFeatures(2),
                         RidgeCV(alphas=np.logspace(-2, 3, 20)))


class WeldModel:
    """One (GP + quadratic ridge) ensemble per output."""

    def __init__(self):
        self.x_scaler = StandardScaler()
        self.gps, self.polys = {}, {}

    def fit(self, X, Y):
        Xs = self.x_scaler.fit_transform(X)
        for j, t in enumerate(TARGETS):
            self.gps[t] = _make_gp(X.shape[1]).fit(Xs, Y[:, j])
            self.polys[t] = _make_poly().fit(Xs, Y[:, j])
        return self

    def predict(self, X):
        """Return mean, std arrays of shape (n, 3).
        std combines the GP's own uncertainty and the disagreement between
        the two sub-models."""
        Xs = self.x_scaler.transform(X)
        means, stds = [], []
        for t in TARGETS:
            g_mu, g_sd = self.gps[t].predict(Xs, return_std=True)
            p_mu = self.polys[t].predict(Xs)
            mu = 0.5 * (g_mu + p_mu)
            sd = np.sqrt(g_sd**2 + (0.5 * (g_mu - p_mu))**2)
            means.append(mu)
            stds.append(sd)
        return np.column_stack(means), np.column_stack(stds)
