
from .ode import *
from ._utils import params_random, params_default

PARAMS_CONFIG = {
    "task": "Lotka_Volterra",
    "env_max": 10,
    "ode_dim": 2,
    "ode_dim_function": 2,
    "params_strategy_list": ["default", "random"],
    "params": {
        "default": params_default,
        "random": params_random,
    },
    "random_params_base": [1.00, 0.30, 3.00, 0.10],
    "default_params_list": [
        [1.00, 0.30, 3.00, 0.10],
        [1.20, 0.39, 2.80, 0.09],
        [1.30, 0.42, 3.20, 0.08],
        [1.10, 0.51, 3.10, 0.11],
        [0.90, 0.39, 2.90, 0.12],
        [0.85, 0.35, 2.85, 0.13],
        [1.15, 0.28, 3.30, 0.14],
        [1.25, 0.40, 3.05, 0.07],
        [0.95, 0.43, 2.65, 0.06],
        [1.40, 0.26, 3.35, 0.15],
    ],
    "random_y0_base": [10.00, 5.00],
    "default_y0_list": [
        [10.00, 5.00],
        [9.60, 4.30],
        [10.10, 5.10],
        [10.95, 4.85],
        [8.90, 6.10],
        [11.10, 5.45],
        [8.75, 5.85],
        [11.40, 4.95],
        [10.05, 5.35],
        [8.85, 5.20],
    ],
    "dt": 0.01,
    "t_min": 0,
    "t_max": 10,
    "curve_names": ["x", "y"],
    "rule_map": ['A->(A+A)', 'A->(A-A)', 'A->(A*A)', 'A->(A/A)', 'A->(A*C)', 
                'A->x', 'A->y'],
    "ntn_list": ['A'],
    "truth_ode_format": ["{0}*x-{1}*x*y", "{3}*x*y-{2}*y"],
    "purification_threshold": 0.05,
}


class LotkaVolterra(ODEDataset):
    params_config = PARAMS_CONFIG
    
    def __init__(self, args):
        super(LotkaVolterra, self).__init__(args, PARAMS_CONFIG)

    def _func(self, x, t, env_id):
        alpha, beta, gamma, delta = iter(self.params[env_id])
        dy = np.asarray([
            alpha * x[0] - beta * x[0] * x[1],
            delta * x[0] * x[1] - gamma * x[1],
        ])
        return dy


if __name__ == "__main__":
    pass
