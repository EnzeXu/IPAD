
from .ode import *
from ._utils import params_random, params_default

PARAMS_CONFIG = {
    "task": "Lorenz",
    "env_max": 5,
    "ode_dim": 3,
    "ode_dim_function": 3,
    "params_strategy_list": ["default", "random"],
    "params": {
        "default": params_default,
        "random": params_random,
    },
    "random_params_base": [28.00, 10.00, 2.67],
    "default_params_list": [
        [28.00, 10.00, 2.67],
        [4.00, 9.90, 4.52],
        [20.00, 5.20, 2.82],
        [10.00, 10.50, 3.08],
        [19.00, 8.90, 6.19],
    ],
    "random_y0_base": [6.00, 6.00, 15.00],
    "default_y0_list": [
        [6.00, 6.00, 15.00],
        [5.00, 7.00, 12.00],
        [5.80, 6.30, 17.00],
        [6.05, 6.40, 14.00],
        [6.25, 6.50, 11.00],
    ],
    "dt": 0.005,  # 0.001
    "t_min": 0,
    "t_max": 5,  # 10
    "curve_names": ["x", "y", "z"],
    "rule_map": ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->C*A',
                'A->x', 'A->y', 'A->z'],
    "ntn_list": ['A'],
    "truth_ode_format": [
        "{1}*y-{1}*x",
        "{0}*x-1.0*x*z-1.0*y",
        "1.0*x*y-{2}*z",
    ],
    "purification_threshold": 0.02,
}


class Lorenz(ODEDataset):
    params_config = PARAMS_CONFIG

    def __init__(self, args):
        super(Lorenz, self).__init__(args, PARAMS_CONFIG)

    def _func(self, x, t, env_id):
        rho, sigma, beta = iter(self.params[env_id])
        dy = np.asarray([
            sigma * (x[1] - x[0]),
            x[0] * (rho - x[2]) - x[1],
            x[0] * x[1] - beta * x[2]
        ])
        return dy


if __name__ == "__main__":
    pass
