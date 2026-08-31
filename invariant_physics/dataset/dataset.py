import argparse

from ._utils import get_now_string, get_partial_mask, load_argparse, print_argparse

from ._lotka_volterra import LotkaVolterra
from ._lorenz import Lorenz
from ._sir import SIR
from ._damped_pendulum import DampedPendulum

ODE_DICT = {
    "Lotka_Volterra": LotkaVolterra,
    "Lorenz": Lorenz,
    "SIR": SIR,
    "Damped_Pendulum": DampedPendulum,
}


def get_dataset(timestring=None):
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--task", type=str, default="Lotka_Volterra", choices=ODE_DICT.keys(), help="ode name")
    # parser.add_argument("--num_env", type=int, default=5, help="number of environments being used for scoring, default 5")
    # parser.add_argument("--seed", type=int, default=0, help="random seed")
    # parser.add_argument("--noise_ratio", type=float, default=0.00, help="noise ratio")
    # parser.add_argument("--sample_strategy", type=str, default="uniform", choices=["uniform", "lhs"], help="sample strategy")
    # parser.add_argument("--params_strategy", type=str, default="default", choices=["default", "random"], help="params strategy")
    # parser.add_argument("--train_test_total", type=str, default="500", help="num_train+num_test. E.g., '500', '500/400/300/200/100'. If you want to specify the total of different environment, use '/' to split")
    # parser.add_argument("--train_ratio", type=float, default=0.80, help="train_ratio")
    # parser.add_argument("--test_ratio", type=float, default=0.10, help="test_ratio")
    # parser.add_argument("--val_ratio", type=float, default=0.10, help="val_ratio")
    # parser.add_argument("--save_figure", type=int, default=0, help="save figure or not")
    # parser.add_argument("--data_dir", type=str, default="data/", help="save_folder")
    # parser.add_argument("--train_sample_strategy", type=str, default="uniform", choices=["uniform", "random"], help="based on the sample strategy in the whole time series, which way to sample train & test set on it")
    #
    # parser.add_argument("--use_new_reward", type=int, default=0, help="0: old score, 1: new score-min, 2: new score-mean")
    # parser.add_argument("--loss_func", type=str, choices=["VF", "L2"], default="L2", help="loss function: L2 or VF")
    # parser.add_argument("--num_run", type=int, default=1, help="num_run")
    # parser.add_argument("--task_ode_num", type=int, default="1", help="ODE # in current task, e.g. for Lotka-Volterra, 1 is for dx, 2 for dy")
    # parser.add_argument("--dataset_sparse", type=str, default="sparse", choices=["sparse", "dense"], help="sparse or dense")
    # parser.add_argument("--dataset_gp", type=int, default=0, choices=[0, 1], help="Gaussian Process or not")
    # parser.add_argument('--main_path', type=str, default="./", help="""directory to the main path""")
    # parser.add_argument('--eta', default=0.99, type=float, help='eta, parsimony coefficient, default 0.99')
    # parser.add_argument('--combine_operator', default='average', type=str, help="""please select which operator used to combine rewards from different environments: [min, average]""")
    # parser.add_argument('--integrate_method', type=str, default="ode_int", choices=["ode_int", "solve_ivp"], help="""integrate_method""")
    #
    # parser.add_argument('--non_ode_sampling', type=str, default="random", choices=["random", "cubic"],
    #                     help="""non_ode_sampling""")
    # parser.add_argument('--shell_timestring', type=str, default="", help="""shell_timestring""")
    #
    # parser.add_argument('--tree_size_strategy', type=str, default="default", choices=["default", "shorten"],
    #                     help="""tree_size_strategy""")
    #
    # parser.add_argument('--n_partial', type=int, default=0,
    #                     help="""n_partial""")
    # parser.add_argument('--n_dynamic', type=int, default=1,
    #                     help="""n_dynamic""")
    # parser.add_argument('--load_data_from_existing', type=int, default=0,
    #                     help="""skip_data_generation""")
    # args = parser.parse_args()
    args, parser = load_argparse(timestring)
    print_argparse(args, parser)

    # assert abs(args.train_ratio + args.test_ratio + args.val_ratio - 1.00) < 1e-3, f"train_ratio ({args.train_ratio}) + test_ratio ({args.test_ratio}) + val_ratio ({args.val_ratio}) should equals to 1.0 !"
    # if not time_string:
    #     args.time_string = get_now_string()
    # else:
    #     args.time_string = time_string
    # args.train_test_total_list = get_train_test_total_list(args.train_test_total, args.num_env, args.seed)
    # args.partial_mask_list = get_partial_mask(args.n_partial, args.num_env, args.seed)
    # print(f"args.train_test_total_list: {type(args.train_test_total_list)}: {args.train_test_total_list}")
    # print(f"str(args.train_test_total_list): {type(str(args.train_test_total_list))}: {str(args.train_test_total_list)}")
    # # print(str(args.train_test_total_list).replace(", ", "/").replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace(",", ""))
    ode = ODE_DICT.get(args.task)(args)
    return ode


if __name__ == "__main__":
    ode = get_dataset()
    ode.build()
