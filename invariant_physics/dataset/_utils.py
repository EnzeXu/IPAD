import random
import os
import sympy as sp
import datetime
import numpy as np
import pandas as pd
import pytz
import re
import pickle
import sys
import json
import argparse
from scipy.stats import qmc
from sympy import symbols, sympify, Mul
from concurrent.futures import ProcessPoolExecutor
from scipy.integrate import odeint
from collections import Counter

def load_argparse(timestring=None):
    parser = argparse.ArgumentParser(description='make_data')
    parser.add_argument(
        '--task',
        default='Lotka_Volterra',
        type=str, choices=['Lotka_Volterra', 'Lorenz', 'SIR', 'Damped_Pendulum'],
        help="benchmark task name, one of [Lotka_Volterra, Lorenz, SIR, Damped_Pendulum]")
    parser.add_argument(
        "--task_ode_num",
        type=int,
        default=1, help="ODE # in current task, e.g. for Lotka_Volterra, 1 is for dx, 2 for dy")
    parser.add_argument(
        '--num_env',
        default=5,
        type=int, help='number of environments being used for scoring, default 5')
    parser.add_argument(
        '--eta',
        default=0.99,
        type=float, help='eta, parsimony coefficient, default 0.99')
    parser.add_argument(
        '--output_dir',
        default='results',
        type=str, help="""directory to store log and Monte Carlo trees""")
    parser.add_argument(
        '--max_added_grammar_count',
        default=6,
        type=int, help='number of grammars can be inserted at once (in forced nodes), 0 means no insertion allowed')
    parser.add_argument(
        "--force",
        type=int,
        default=0, help="whether to force simplified nodes back into the tree or not")
    parser.add_argument(
        "--use_new_reward",
        type=int,
        default=0, help="0: old score, 1: new score-min, 2: new score-mean")
    parser.add_argument(
        "--reward_rescale",
        type=int,
        default=0, help="whether or not to use rescale in the reward function")
    parser.add_argument(
        '--data_dir',
        default='data/',
        type=str, help="""directory to datasets""")
    parser.add_argument(
        '--error_tolerance',
        default=0.99,
        type=float, help='error_tolerance for reward functions 1 and 2, default 0.99')
    parser.add_argument(
        "--num_run",
        type=int,
        default=1, help="Number of tests to run")
    parser.add_argument(
        "--transplant_step",
        type=int,
        default=500, help="Number of MCTS iterations per transplant")
    parser.add_argument(
        "--num_transplant",
        type=int,
        default=1, help="Number of transplant")
    parser.add_argument(
        "--loss_func",
        type=str,
        choices=["VF", "L2"],
        default="L2", help="loss function: L2 or VF")
    parser.add_argument(
        '--combine_operator',
        default='average',
        type=str, help="""please select which operator used to combine rewards from different environments: 
                          [min
                           average, 
                          ]""")
    parser.add_argument(
        '--min_lam_diff',
        default=0.,
        type=float, help='minimum weight for difference/residual reward in reward_function 2, default 0.0')
    parser.add_argument(
        "--seed",
        type=int,
        default=0, help="Random seed used")
    parser.add_argument("--noise_ratio", type=float, default=0.00, help="noise ratio")
    parser.add_argument("--resume", type=int, default=0, help="resume (1) or not (0)")
    parser.add_argument("--params_strategy", type=str, default="default", choices=["default", "random"], help="params strategy")

    parser.add_argument("--n_data_samples", type=int, default=None,
                        help="number of data samples")
    parser.add_argument('--n_dynamic', type=str, default="150",
                        help="""n_dynamic""")
    parser.add_argument("--dataset_sparse", type=str, default="sparse", choices=["sparse", "dense"], help="sparse or dense")
    parser.add_argument("--save_figure", type=int, default=0, help="save figure or not")
    parser.add_argument("--dataset_gp", type=int, default=0, choices=[0, 1], help="Gaussian Process or not")
    parser.add_argument('--main_path', type=str, default="./", help="""directory to the main path""")
    parser.add_argument('--integrate_method', type=str, default="ode_int", choices=["ode_int", "solve_ivp"],
                        help="""integrate_method""")
    parser.add_argument("--train_ratio", type=float, default=0.80, help="train_ratio")
    parser.add_argument("--test_ratio", type=float, default=0.10, help="test_ratio")
    parser.add_argument("--val_ratio", type=float, default=0.10, help="val_ratio")
    # parser.add_argument("--dataset_sparse", type=str, default="sparse", choices=["sparse", "dense"],
    #                     help="sparse or dense")
    parser.add_argument('--non_ode_sampling', type=str, default="random", choices=["random", "cubic"],
                        help="""non_ode_sampling""")
    parser.add_argument('--timestring', type=str, default="", help="""timestring""")
    parser.add_argument('--tree_size_strategy', type=str, default="default", choices=["default", "shorten"],
                        help="""tree_size_strategy""")
    parser.add_argument('--n_partial', type=int, default=0,
                        help="""n_partial""")
    parser.add_argument("--train_sample_strategy", type=str, default="uniform", choices=["uniform", "random"], help="based on the sample strategy in the whole time series, which way to sample train & test set on it")

    parser.add_argument('--load_data_from_existing', type=int, default=0,
                        help="""skip_data_generation""")
    parser.add_argument("--sample_strategy", type=str, default="uniform", choices=["uniform", "lhs"],
                        help="sample strategy")
    parser.add_argument("--select_run_strategy", type=str, default="highest_reward", choices=["highest_reward", "most_frequent"],
                        help="select_run_strategy")
    parser.add_argument("--extract_csv", type=int, default=0,
                        choices=[0, 1],
                        help="extract_csv")
    parser.add_argument("--env_id", type=int, default=None,
                        help="env_id. SPL only")
    parser.add_argument("--record_task_date", type=str, default="00000000",
                        help="record_task_date. SPL only")
    parser.add_argument("--vf_func_num", type=int, default=50,
                        help="vf_func_num")
    parser.add_argument("--swap_n_dynamic_list", type=int, default=1,
                        help="swap_n_dynamic_list")


    args = parser.parse_args()
    assert abs(args.train_ratio + args.test_ratio + args.val_ratio - 1.00) < 1e-3, f"train_ratio ({args.train_ratio}) + test_ratio ({args.test_ratio}) + val_ratio ({args.val_ratio}) should equals to 1.0 !"
    if not args.timestring or len(args.timestring) < 1:
        if not timestring:
            args.timestring = get_now_string()
        else:
            args.timestring = timestring
    assert len(args.timestring) == 22

    # args.n_data_samples_list = get_n_data_samples_list(args.n_data_samples, args.num_env, args.seed)
    args.n_dynamic_list = get_n_dynamic_list(args.n_dynamic, args.num_env, args.seed, swap_n_dynamic_list=args.swap_n_dynamic_list)
    args.partial_mask_list = get_partial_mask(args.n_partial, args.num_env, args.seed)
    return args, parser

def print_argparse(args, parser):
    args_dict = vars(args)
    filtered_sorted_args_dict = {key: args_dict[key] for key in sorted(args_dict) if
                                 key in parser._option_string_actions}
    print(json.dumps(filtered_sorted_args_dict, indent=4))


def sample_lhs(lb, ub, n, skip=1):
    large_n = skip * n
    sampler = qmc.LatinHypercube(d=1)
    sample = sampler.random(n=large_n)
    sample = qmc.scale(sample, lb, ub)
    sample = np.sort(np.squeeze(sample))
    sample = sample[::skip]
    return sample


def params_default(num_env, **kwargs):
    default_list = kwargs["default_list"]
    return default_list[:num_env]


def params_random(num_env, seed_offset=0, **kwargs):
    base, random_rate, seed = kwargs["base"], kwargs["random_rate"], kwargs["seed"]
    # params = []
    random.seed(seed + seed_offset)
    params = [round(random.uniform((1 - random_rate) * item, (1 + random_rate) * item), 7) for item in base]
    return params



def save_to_csv(save_path, cols, headers=None):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    if headers:
        assert len(cols) == len(headers), f"len(cols) != len(headers): {len(cols)} != {len(headers)}"
        df = pd.DataFrame({f"{headers[i]}": cols[i] for i in range(len(cols))})
        df.to_csv(save_path, index=False)
    else:
        df = pd.DataFrame({f"item {i}": cols[i] for i in range(len(cols))})
        df.to_csv(save_path, index=False, header=False)


def load_data(data_path):
    with open(data_path, "rb") as f:
        data_dic = pickle.load(f)
    assert "data_train" in data_dic
    assert "data_val" in data_dic
    assert "data_test" in data_dic
    return (
        data_dic,
        [data_dic["data_train"]["y_noise"], data_dic["data_train"]["dy_noise"]],
        [data_dic["data_val"]["y_noise"], data_dic["data_val"]["dy_noise"]],
        [data_dic["data_test"]["y_noise"], data_dic["data_test"]["dy_noise"]],
    )



def get_now_string(time_string="%Y%m%d_%H%M%S_%f"):
    # return datetime.datetime.now().strftime(time_string)
    est = pytz.timezone('America/New_York')

    # Get the current time in UTC and convert it to EST
    utc_now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
    est_now = utc_now.astimezone(est)

    # Return the time in the desired format
    return est_now.strftime(time_string)


def is_term_constant(term):
    return isinstance(term, (sp.Float, sp.Integer, sp.Rational))


def score_match_terms(term_list_1: list, term_list_2: list):
    term_list_1 = list(set(term_list_1))
    term_list_2 = list(set(term_list_2))
    match_cnt = sum([int(item in term_list_2) for item in term_list_1])
    return match_cnt / max(len(term_list_1), len(term_list_2))


def extract(expression):
    """
    Extract the `full_terms`, `terms`, and `coefficient_terms` (all in list) of an expression
    The order of each part is by str comparison of items in `terms`
    :param expression: e.g., "3*y+2*sin(x)-3*x**2+2*x*y"
    :return: [2*sin(x), -3*x**2, 2*x*y, 3*y], [sin(x), x**2, x*y, y], [2, -3, 2, 3]
    """
    expr = sp.sympify(expression)
    raw_terms = list(sp.Add.make_args(expr))
    # print(raw_terms)
    result = []

    for term in raw_terms:
        term = sp.sympify(term)
        # if is_term_constant(term):
        #     continue
        # print(sp.Mul.make_args(term))
        coefficient_list = []
        non_coefficient_list = []
        for factor in sp.Mul.make_args(term):
            if is_term_constant(factor):
                coefficient_list.append(factor)
            else:
                non_coefficient_list.append(factor)
            # print(f"{factor}: {is_term_constant(factor)}")
        coefficient_part = sp.sympify(sp.Mul(*[sp.sympify(item) for item in coefficient_list]))
        non_coefficient_part = sp.sympify(sp.Mul(*[sp.sympify(item) for item in non_coefficient_list]))
        result.append([term, non_coefficient_part, coefficient_part])

    result = sorted(result, key=lambda x: str(x[1]))
    full_terms = [item[0] for item in result]
    terms = [item[1] for item in result]
    coefficient_terms = [item[2] for item in result]
    return full_terms, terms, coefficient_terms

# Sequential approach
def evaluate_expression(expression_str, variable_list, variable_values):
    # print("debug:", expression_str, variable_list, variable_values, f"variable_values shape:{variable_values.shape}")
    variables = symbols(variable_list)
    expr = sympify(expression_str)
    results = []
    for values in variable_values:
        # variable_dict = {var: val for var, val in zip(variables, values)}
        variable_dict = {var: float(val) if isinstance(val, np.float64) else val for var, val in
                         zip(variables, values)}
        result = expr.subs(variable_dict).evalf()
        results.append(result)
    results = np.array(results, dtype=float)
    # variable_dict = {var: val for var, val in zip(variables, variable_values)}
    # result = expr.subs(variable_dict)
    # result_value = result.evalf()
    # return result_value
    # print("one evaluate_expression finished")
    return np.mean(results)


# Parallel approach
# def evaluate_expression(expression_str, variable_list, variable_values):
#     variables = symbols(variable_list)
#     expr = sympify(expression_str)
#
#     def evaluate_single(values):
#         variable_dict = {var: val for var, val in zip(variables, values)}
#         return expr.subs(variable_dict).evalf()
#
#     with ProcessPoolExecutor() as executor:
#         results = list(executor.map(evaluate_single, variable_values))
#     print("one evaluate_expression finished")
#     return np.mean(results)

def remove_constant(one_list):
    return [item for item in one_list if item != "1"]


def determine_most_frequent_terms(input_list):
    assert len(input_list) > 0
    dic = dict()
    vals = input_list
    for item in vals:
        sub_vals = item.values()
        for sub_item in sub_vals:
            if sub_item["purified_predicted_terms"] not in dic:
                dic[sub_item["purified_predicted_terms"]] = [sub_item["purified_predicted_terms"], 1]
            else:
                dic[sub_item["purified_predicted_terms"]][1] += 1
    sorted_values = sorted(list(dic.values()), key=lambda x: -x[1])
    print(sorted_values)
    return sorted_values[0][0]


def transform_sympy(input_eq_str):
    expr = sympify(input_eq_str)
    C = symbols('C')
    transformed_terms = []
    for term in expr.as_ordered_terms():
        coeff, rest = term.as_coeff_Mul()
        if rest == 1:
            transformed_terms.append(C)
        else:
            transformed_terms.append(Mul(C, rest))
    transformed_expr = sum(transformed_terms)
    return str(transformed_expr)

def set_eq_precision(eq_str, digit):
    expr = sympify(eq_str)
    C = symbols('C')
    transformed_terms = []
    for term in expr.as_ordered_terms():
        coeff, rest = term.as_coeff_Mul()
        rounded_coeff = round(coeff, digit)
        if rest == 1:
            transformed_terms.append(rounded_coeff)
        else:
            transformed_terms.append(rounded_coeff * rest)
    transformed_expr = sum(transformed_terms)
    return str(transformed_expr)

math_functions = {
    'sin': "np.sin",
    'cos': "np.cos",
    'exp': "np.exp",
    'log': "np.log",
}

def math_enc(eq):
    for one_key in math_functions:
        eq = eq.replace(one_key, math_functions[one_key])
    return eq

def math_dec(eq):
    for one_key in math_functions:
        eq = eq.replace(math_functions[one_key], one_key)
    return eq

def evaluate_eq_into_value(eq_str, curve_names, data_points):
    n_dynamic = data_points.shape[0]
    results = np.zeros([n_dynamic, data_points.shape[1]])
    eval_context = {'np': np}
    for i_dynamic in range(n_dynamic):
        for i, point in enumerate(data_points[i_dynamic]):
            # print(f"curve_names: {curve_names}")
            # print(f"point: {point}")
            var_values = {var_name: value for var_name, value in zip(curve_names, point)}
            try:
                var_values.update(eval_context)
                # print(f"math_enc(eq_str): {math_enc(eq_str)}")
                # print(f"var_values: {var_values}")
                result = eval(math_enc(eq_str), {"__builtins__": None}, var_values)
            except Exception as e:
                print(f"Error in {math_enc(eq_str)}:", e)
            results[i_dynamic][i] = result
    return results


def evaluate_trajectory_rmse(ode, eq_str, i_env, task_ode_num):
    # data_points = ode.y_noise[i_env][ode.test_indices_list[i_env]]
    data_points = ode.y_noise[i_env]
    dy_prediction = evaluate_eq_into_value(eq_str, ode.params_config["curve_names"], data_points)
    # dy_truth = ode.dy_noise[i_env][ode.test_indices_list[i_env], task_ode_num - 1]
    dy_truth = ode.dy_noise[i_env][:, :, task_ode_num - 1]
    # print(f"dy_prediction shape: {dy_prediction.shape} dy_truth shape: {dy_truth.shape}")
    assert dy_prediction.shape == dy_truth.shape
    mse = np.mean((dy_prediction - dy_truth) ** 2)
    rmse = np.sqrt(mse)
    variance = np.var(dy_truth)
    std_deviation = np.sqrt(variance)
    relative_mse = mse / variance if variance != 0 else float('inf')
    relative_rmse = rmse / std_deviation if std_deviation != 0 else float('inf')
    return mse, rmse, relative_mse, relative_rmse


# def get_n_data_samples_list(n_data_samples: str, num_env: int, seed=None):
#     """
#     Args:
#         n_data_samples: supports three types of argument:
#             (1) one integer, like 500, indicating it's a balanced dataset;
#             (2) integers split by "/", like 500/400/300/200/100. There would be an error if its length mismatches the num_env;
#             (3) a string "default_x", following a built-in dict as below.
#         num_env: an integer, number of environment
#         seed: an integer for generating random ordering list
#     Returns:
#         a list of environment size, e.g., [500, 400, 300, 200, 100].
#     """
#     default_dic = {
#         "default_0": "500/500/500/500/500",
#         "default_1": "20/20/20/20/20",
#         "default_2": "100/100/100/100/100",
#         "default_3": "500/400/40/20/10",
#         "default_4": "500/80/40/20/10",
#         "default_5": "100/50/50/20/10",
#         "default_6": "500/400/300/200/100",
#         "default_7": "500/400/300/200/10",
#         "default_8": "500/400/300/20/10",
#         "default_10": "200/200/200/200/200",
#         "default_11": "500/400/40/40/20",
#         "default_12": "500/200/200/50/50",
#         "default_13": "500/125/125/125/125",
#         "default_15": "1000/1000/1000/1000/1000",
#         "default_16": "900/900/900/900/900",
#         "default_17": "800/800/800/800/800",
#         "default_18": "700/700/700/700/700",
#         "default_19": "600/600/600/600/600",
#         "default_20": "500/500/500/500/500",
#         "default_21": "400/400/400/400/400",
#         "default_22": "300/300/300/300/300",
#         "default_30": "25/25/25/25/25",
#         "default_31": "50/50/50/50/50",
#         "default_32": "75/75/75/75/75",
#         "default_33": "100/100/100/100/100",
#         "default_34": "200/200/200/200/200",
#         "default_35": "500/500/500/500/500",
#         "default_36": "400/400/400/400/400",
#         "default_37": "800/800/800/800/800",
#         "default_38": "1600/1600/1600/1600/1600",
#         "default_41": "400/400/400/400/400",
#         "default_43": "1600/1600/1600/1600/1600",
#         "default_45": "6400/6400/6400/6400/6400",
#         "default_47": "25600/25600/25600/25600/25600",
#     }
#
#     if n_data_samples.isdigit():
#         n_data_samples_list = [int(n_data_samples) for i in range(num_env)]
#     else:
#         if "/" not in n_data_samples:
#             assert n_data_samples in default_dic, "Error: key error in get_n_data_samples_list: " + str(n_data_samples)
#             string = default_dic[n_data_samples]
#         else:
#             string = n_data_samples
#         parts = string.split("/")
#         n_data_samples_list = [int(item) for item in parts]
#     assert len(n_data_samples_list) == num_env, "Error: mismatching between " + str(n_data_samples) + " and " + str(num_env)
#     one_order = generate_random_order(num_env, seed)
#     n_data_samples_list_swapped = reseat(n_data_samples_list, one_order)
#     print(f"Swap n_data_samples size: {n_data_samples_list} -> {n_data_samples_list_swapped}")
#     return n_data_samples_list_swapped


def get_n_dynamic_list(n_dynamic: str, num_env: int, seed=None, swap_n_dynamic_list=1):
    """
    Args:
        n_dynamic: supports three types of argument:
            (1) one integer, like 500, indicating it's a balanced dataset;
            (2) integers split by "/", like 500/400/300/200/100. There would be an error if its length mismatches the num_env;
            (3) a string "default_x", following a built-in dict as below.
        num_env: an integer, number of environment
        seed: an integer for generating random ordering list
    Returns:
        a list of environment size, e.g., [500, 400, 300, 200, 100].
    """
    default_dic = {
        "default_0": "10/10/10/10/10",
        "default_1": "40/40/40/40/40",
        "default_2": "120/20/20/20/20",
        "default_3": "120/30/30/10/10",
    }
    if n_dynamic.isdigit():
        n_dynamic_list = [int(n_dynamic) for i in range(num_env)]
    else:
        if "/" not in n_dynamic:
            assert n_dynamic in default_dic, "Error: key error in get_n_dynamic_list: " + str(n_dynamic)
            string = default_dic[n_dynamic]
        else:
            string = n_dynamic
        parts = string.split("/")
        n_dynamic_list = [int(item) for item in parts]
    assert len(n_dynamic_list) == num_env, "Error: mismatching between " + str(n_dynamic) + " and " + str(num_env)
    if swap_n_dynamic_list:
        one_order = generate_random_order(num_env, seed)
        n_dynamic_list_swapped = reseat(n_dynamic_list, one_order)
    else:
        n_dynamic_list_swapped = n_dynamic_list
    # print(f"Swap n_dynamic size: {n_dynamic_list} -> {n_dynamic_list_swapped}")
    return n_dynamic_list_swapped

def get_partial_mask(n_partial: int, num_env: int, seed=None):
    assert n_partial in list(range(0, num_env + 1))
    base = [0.00] * n_partial + [1.00] * (num_env - n_partial)
    one_order = generate_random_order(num_env, seed)
    base_swapped = reseat(base, one_order)
    if n_partial > 0:
        print(f"Swap param partial mask: {base} -> {base_swapped}")
    return base_swapped


def generate_random_order(n, seed=None):
    if seed is not None:
        random.seed(seed)
    numbers = list(range(n))
    random.shuffle(numbers)
    return numbers


def reseat(one_list, one_order):
    assert len(one_list) == len(one_order)
    return [one_list[one_order[i]] for i in range(len(one_order))]


def calculate_parameter_rmse(eq_list_1, eq_list_2):
    assert len(eq_list_1) == len(eq_list_2), "len(eq_list_1) != len(eq_list_2)"
    n = len(eq_list_1)
    sum_rmse = 0.00

    for eq_id in range(n):
        full_terms1, terms1, coefficient_terms1 = extract(eq_list_1[eq_id])
        full_terms2, terms2, coefficient_terms2 = extract(eq_list_2[eq_id])
        # print(f"coefficient_terms1={coefficient_terms1}, coefficient_terms2={coefficient_terms2}")
        one_rmse = rmse(coefficient_terms1, coefficient_terms2)
        print(f"calculate_parameter_rmse: (truth) {coefficient_terms1} vs. (prediction) {coefficient_terms2}: rmse={one_rmse:.6f}")
        sum_rmse += one_rmse

    avg_rmse = sum_rmse / n
    return avg_rmse


def rmse(list1, list2):
    residual = np.asarray(list1).astype(float) - np.asarray(list2).astype(float)
    return np.sqrt(np.sum(residual ** 2))


# def simplify_and_replace_constants(expr_str):
#     """
#     e.g.
#     Args:
#         expr_str:
#
#     Returns:
#
#     """
#     # Define the symbols
#     # x, y = sp.symbols('x y')
#     C = sp.Symbol('C')
#
#
#     # Parse the expression into a SymPy expression
#     parsed_expr = sp.sympify(expr_str)
#
#     # Simplify the expression
#     simplified_expr = parsed_expr
#     # print(f"simplified_expr: {simplified_expr}")
#
#     # Find all constants in the expression
#     constants = [term for term in simplified_expr.atoms(sp.Number)]
#     # print(constants)
#
#     # Replace each constant with 'C'
#     for const in constants:
#         simplified_expr = simplified_expr.subs(const, C)
#
#     return str(simplified_expr)


def most_common(lst):
    if not lst:
        return None
    count = Counter(lst)
    max_count = max(count.values())
    for item in lst:
        if count[item] == max_count:
            return item


def simplify_and_replace_constants(expr_str):
    """
    Replaces all float numbers in the equation string with 'C', except when they
    appear as power indices (e.g., in x ** 2).

    Args:
        expr_str (str): The input equation as a string.

    Returns:
        str: The equation with float numbers replaced by 'C'.
    """
    # Regex pattern to match floats not followed by '**' (to avoid replacing exponents)
    expr_str = str(sp.sympify(expr_str))
    pattern = r'(?<!\*\*)\b\d+\.\d+\b'

    # Replace matched floats with 'C'
    modified_expr = re.sub(pattern, 'C', expr_str)
    modified_expr = modified_expr.replace("-", "+")
    if modified_expr[0] == "+":
        return modified_expr[1:]

    return modified_expr


def generate_ordered_indices(total_size, train_size, val_size, test_size):
    assert train_size + val_size + test_size == total_size, f"{train_size} + {val_size} + {test_size} == {total_size}"

    indices = np.random.permutation(total_size)

    train_index = np.sort(indices[:train_size])
    val_index = np.sort(indices[train_size:train_size + val_size])
    test_index = np.sort(indices[train_size + val_size:])

    return train_index, val_index, test_index


def judge_expression_equal(str1, str2):
    # print(f"str1 = {str1}")
    # print(f"str2 = {str2}")
    str1 = str1.replace("C", "1")
    str2 = str2.replace("C", "1")
    full_terms1, terms1, coefficient_terms1 = extract(str1)
    full_terms2, terms2, coefficient_terms2 = extract(str2)

    one_term = sp.simplify("1")

    if one_term in terms1:
        terms1.remove(one_term)
    if one_term in terms2:
        terms2.remove(one_term)

    # print(full_terms1, terms1, coefficient_terms1)
    # print(full_terms2, terms2, coefficient_terms2)

    return int(str(terms1) == str(terms2))


# To skip completed tasks
def generate_record_csv(ode_name):
    end_file_path = f"logs/summary/logs_{ode_name}_end.csv"
    assert os.path.exists(end_file_path)
    df = pd.read_csv(end_file_path)
    df = df[["start_time", "n_dynamic", "noise_ratio", "task_ode_num", "env_id", "seed"]]
    df = df.sort_values(by=["n_dynamic", "noise_ratio", "task_ode_num", "env_id", "seed"])
    df = df.reset_index(drop=True)
    # print(df)
    return df


# To skip completed tasks
def check_existing_record(task_date, ode_name, n_dynamic, noise_ratio, task_ode_num, env_id, seed):
    record_folder = f"logs/{ode_name}_record/"
    if not os.path.exists(record_folder):
        os.makedirs(record_folder)
    record_path = f"{record_folder}/{task_date}_record.csv"
    if os.path.exists(record_path):
        df = pd.read_csv(record_path)
        print(f"Loaded the record csv from {record_path}.", file=sys.stderr)
    else:
        df = generate_record_csv(ode_name)
        df.to_csv(record_path, index=False)
        print(f"Not found. Generate and save the record csv to {record_path}.", file=sys.stderr)
    df = df[
        (df['n_dynamic'] == n_dynamic) &
        (df['noise_ratio'].astype(float).round(3) == round(float(noise_ratio), 3)) &
        (df['task_ode_num'].astype(int) == int(task_ode_num)) &
        (df['env_id'].astype(int) == int(env_id)) &
        (df['seed'].astype(int) == int(seed))
        ]
    # print(len(df))
    if len(df) >= 1:
        return True
    return False


if __name__ == "__main__":


    result1 = judge_expression_equal("C*x*y + C*x", "C*x*y + x")
    result2 = judge_expression_equal("C*sin(x) + C*y", "C*sin(C*x) + C*y")
    result3 = judge_expression_equal("C*x*y + C*x", "C*x*y + x + C")
    result4 = judge_expression_equal("C*x*y + C*x", "C+x+C*x*y")

    print(result1)
    print(result2)
    print(result3)

    # print(simplify_and_replace_constants("-8.7 / 1.5 * sin(x) - 1.1 * y"))
    # print(simplify_and_replace_constants("-8.7 / 1.5 * sin(x) - 1.1 * y ** 2"))

    # print(most_common(["a", 1, "a", "b", "b", 123, 2.31]))


    # eq_list_1 =  ['1.0*x-0.3*x*y', '1.2*x-0.39*x*y', '1.3*x-0.42*x*y', '1.1*x-0.51*x*y', '0.9*x-0.39*x*y']
    #
    # eq_list_2 = ['-0.299953343018*x*y + 0.999846119139*x', '-0.389932529684*x*y + 1.199794591706*x', '-0.419885185789*x*y + 1.299649068268*x', '-0.509905116574*x*y + 1.099800403716*x', '-0.389934487886*x*y + 0.899851398684*x']
    #
    #
    # print(calculate_parameter_rmse(eq_list_1, eq_list_2))
    #
    # print(rmse([9.81000000000000], [9.81000000000000]))


    # get_train_test_total_list("default_1", 5)
    # numbers = [10,20,30,0,50]
    # order = generate_random_order(5, 500)
    # print(order)
    # print(reseat(numbers, order))
    # a = " -0.389927009553766*x*y + 1.1997924353311362*x"
    # a = "-1/(5*x + 1) * (1+2*x) - 1*x*(1-y)"
    # print(transform_sympy(a))
    # a = {
    #     "1": {
    #         "purified_predicted_terms": "aaaa"
    #     },
    #     "2": {
    #         "purified_predicted_terms": "bbb"
    #     },
    #     "3": {
    #         "purified_predicted_terms": "aaaa"
    #     },
    #     "4": {
    #         "purified_predicted_terms": "c"
    #     },
    # }
    # print(determine_most_frequent_terms(a))

    # expression = "x+y+z"
    # variables = ["x", "y", "z"]
    # values = [1.1, 2.0, -1.4]
    # print(evaluate_expression(expression, variables, values))

    # expr1 = "3*y+2*sin(x)-3*x**2+2.1*x*y"
    # expr2 = "15*x**2*y/(x + 1) - 13*x*2.0*y/(x + 25) + 5.1*x/(x + 5.0)"
    # parts = extract(expr1)
    # print(parts)
    # # print(score_match_terms(parts[1], parts[1]))
    # print(get_partial_mask(3, 5, 888))

    pass
