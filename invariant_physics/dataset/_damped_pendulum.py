
from .ode import *
from ._utils import params_random, params_default

PI = 3.14159_26535_89793_23846_26433_83279_50288_41971_69399_37510

PARAMS_CONFIG = {
    "task": "Damped_Pendulum",
    "env_max": 5,
    "ode_dim": 2,
    "ode_dim_function": 2,
    "params_strategy_list": ["default", "random"],
    "params": {
        "default": params_default,
        "random": params_random,
    },
    "random_params_base": [1.00, 0.0],
    "default_params_list": [
        [1.00, 1.5],
        [0.90, 1.7],
        [0.80, 1.9],
        [1.10, 2.1],
        [1.20, 2.3],
    ],
    "random_y0_base": [0.25 * PI, 0.00],
    "default_y0_list": [
        [0.25 * PI, 0.00],
        [-0.25 * PI, 0.50],
        [0.5 * PI, 1.00],
        [-0.5 * PI, -0.50],
        [0.00 * PI, -1.00],
    ],
    "dt": 0.01,
    "t_min": 0,
    "t_max": 10,
    "curve_names": ["x", "y"],
    # "rule_map": ['A->(A+A)', 'A->(A-A)', 'A->x', 'A->y', 'A->-10*sin(x)'],
    # "rule_map": ['A->(A+A)', 'A->(A-A)', 'A->(A*A)', 'A->(A/A)', 'A->(A*C)',
    #                 'A->x', 'A->y', 'A->sin(x)'],
    "rule_map": ['A->(A+A)', 'A->(A-A)', 'A->(A*A)', 'A->(A/A)', 'A->(A*C)',
                 'A->x', 'A->y', 'A->sin(B)', 'A->cos(B)', 'A->log(B)', 'B->(B*C)', 'B->x', 'B->y'],
    "ntn_list": ['A', 'B'],
    "truth_ode_format":  ["1.0 * y", "-10 / {0} * sin(x) - {1} * y"], # "truth_ode_format": ["y", "-10 / {0} * sin(x) - {1} * y"],
    "purification_threshold": 0.05,
}


class DampedPendulum(ODEDataset):
    params_config = PARAMS_CONFIG

    def __init__(self, args):
        super(DampedPendulum, self).__init__(args, PARAMS_CONFIG)

    def _func(self, x, t, env_id):
        # alpha, beta, gamma, delta = iter(self.params[env_id])
        l, bm = iter(self.params[env_id])
        g = 10.00
        dy = np.asarray([
            x[1],
            -g / l * np.sin(x[0]) - bm * x[1], #g / l * np.cos(x[0]) - bm * x[1], # -g / l * np.sin(x[0]) - bm * x[1],
        ])
        return dy


if __name__ == "__main__":
    pass
