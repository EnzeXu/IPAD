import signal
from contextlib import contextmanager
import numpy as np
import sympy as sp
from concurrent.futures import ProcessPoolExecutor

from ..dataset import extract, evaluate_expression

class TimeOutException(Exception):
    def __init__(self, msg=''):
        self.msg = msg

@contextmanager
def time_limit(seconds, msg=''):
    def _handler(signum, frame):
        raise TimeOutException("Timed out for operation {}".format(msg))
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(seconds))
    try:
        yield
    except TimeOutException:
        raise 
    finally:
        signal.alarm(0)

def purify_strategy(eq, data, variable_list, threshold=0.05, traj_jump=20):
    # data is in shape (N, m). Here m is the dimension of the ODE system
    full_terms, terms, _ = extract(eq)

    data_y_noise = data[0]
    data_y_noise = [item[::traj_jump] for item in data_y_noise]
    n_env = len(data_y_noise)

    abs_value_array = np.zeros([n_env, len(full_terms)])
    abs_ratio_array = np.zeros([n_env, len(full_terms)])
    for i in range(n_env):
        one_env_n_dynamic = len(data_y_noise[i])
        for j, one_full_term in enumerate(full_terms):
            for i_dynamic in range(len(data_y_noise[i])):
                abs_value_array[i][j] += np.abs(evaluate_expression(one_full_term, variable_list, data_y_noise[i][i_dynamic])) / one_env_n_dynamic
        for j in range(len(full_terms)):
            abs_ratio_array[i][j] = abs_value_array[i][j] / np.sum(abs_value_array[i])
    avg_ratio = np.average(abs_ratio_array, axis=0)
    purified_full_terms = [full_terms[i] for i in range(len(full_terms)) if avg_ratio[i] >= threshold]
    purified_eq = sp.sympify(sp.Add(*purified_full_terms))
    return purified_eq, avg_ratio, full_terms, terms


def evaluate_for_env_parallel(i, full_terms, data_y_noise, variable_list):
    abs_value_array = np.zeros(len(full_terms))
    abs_ratio_array = np.zeros(len(full_terms))
    one_env_n_dynamic = len(data_y_noise[i])

    for j, one_full_term in enumerate(full_terms):
        for i_dynamic in range(len(data_y_noise[i])):
            abs_value_array[j] += np.abs(
                evaluate_expression(one_full_term, variable_list, data_y_noise[i][i_dynamic])) / one_env_n_dynamic
    for j in range(len(full_terms)):
        abs_ratio_array[j] = abs_value_array[j] / np.sum(abs_value_array)

    return abs_ratio_array


def purify_strategy_parallel(eq, data, variable_list, threshold=0.05, traj_jump=20):
    full_terms, terms, _ = extract(eq)
    data_y_noise = data[0]
    data_y_noise = [item[::traj_jump] for item in data_y_noise]
    n_env = len(data_y_noise)

    # Use ProcessPoolExecutor to parallelize the task of calculating abs_value_array and abs_ratio_array for each environment
    with ProcessPoolExecutor() as executor:
        abs_ratio_array = list(
            executor.map(evaluate_for_env_parallel, range(n_env), [full_terms] * n_env, [data_y_noise] * n_env,
                         [variable_list] * n_env))

    avg_ratio = np.average(abs_ratio_array, axis=0)
    purified_full_terms = [full_terms[i] for i in range(len(full_terms)) if avg_ratio[i] >= threshold]
    purified_eq = sp.sympify(sp.Add(*purified_full_terms))

    return purified_eq, avg_ratio, full_terms, terms

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
