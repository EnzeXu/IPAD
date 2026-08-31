
from .ode import *
from ._utils import params_random, params_default

PARAMS_CONFIG = {
    "task": "SIR",
    "env_max": 10,
    "ode_dim": 3,
    "ode_dim_function": 3,
    "params_strategy_list": ["default", "random"],
    "params": {
        "default": params_default,
        "random": params_random,
    },
    "random_params_base": [0.010, 0.050],
    "default_params_list": [
        [0.010, 0.050],
        [0.011, 0.040],
        [0.012, 0.043],
        [0.013, 0.045],
        [0.014, 0.047],
        [0.009, 0.060],
        [0.008, 0.058],
        [0.007, 0.056],
        [0.014, 0.054],
        [0.015, 0.052],
    ],
    "random_y0_base": [50.0, 40.0, 10.0],
    "default_y0_list": [
        [50.0, 40.0, 10.0],
        [50.5, 41.0, 8.5],
        [55.0, 40.0, 5.0],
        [47.5, 48.5, 4.0],
        [48.0, 49.0, 3.0],
        [43.0, 51.0, 6.0],
        [47.0, 46.0, 7.0],
        [48.5, 43.5, 8.0],
        [49.0, 42.0, 9.0],
        [30.0, 55.0, 15.0],
    ],
    "dt": 0.05,
    "t_min": 0,
    "t_max": 50,
    "curve_names": ["x", "y", "z"],
    "rule_map": ['A->(A+A)', 'A->(A-A)', 'A->(A*A)', 'A->(A*C)',
                'A->x', 'A->y','A->z'],
    "ntn_list": ['A'],
    "truth_ode_format": [
        "-{0}*x*y",
        "{0}*x*y-{1}*y",
        "{1}*y",
    ],
    "purification_threshold": 0.20,
}


class SIR(ODEDataset):
    params_config = PARAMS_CONFIG

    def __init__(self, args):
        super(SIR, self).__init__(args, PARAMS_CONFIG)

    def _func(self, x, t, env_id):
        beta, gamma = iter(self.params[env_id])
        dy = np.asarray([
            - beta * x[0] * x[1],
            beta * x[0] * x[1] - gamma * x[1],
            gamma * x[1]
        ])
        return dy


if __name__ == "__main__":
    pass
