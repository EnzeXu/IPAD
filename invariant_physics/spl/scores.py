import numpy as np
from functools import partial

from scipy.optimize import minimize

from ._utils import time_limit
from .production_rule_utils import simplify_eqs

from ..loss import VF_Loss
from ..dataset import extract

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

def combine_rewards_original(r_diff_list, r_parsimony_list, 
                            error_tolerance=None,
                            combine_operator="min",
                            num_samples=None):    
    # SPL, reward 0
    if combine_operator == "min":
        combine_operator = min
    elif combine_operator == "average" or combine_operator == "mean":
        combine_operator = partial(np.average, weights=num_samples)
    elif combine_operator == "average_pure":
        combine_operator = np.average
    return combine_operator(
        [rd*rp 
         for rd, rp in zip(r_diff_list, r_parsimony_list)]
    )

def combine_rewards_epsilon_piecewise(r_diff_list, r_parsimony_list, 
                                      error_tolerance=0.99,
                                      combine_operator="min"):
    # reward 1
    if combine_operator == "min":
        combine_operator = min
    elif combine_operator == "average" or combine_operator == "mean":
        combine_operator = np.mean
    return combine_operator(
        [0.5*rd if rd < error_tolerance else 0.5*rd + 0.5*rp
         for rd, rp in zip(r_diff_list, r_parsimony_list)]
    )

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
def combine_rewards_epsilon_sigmoid(r_diff_list, r_parsimony_list, 
                                    error_tolerance=0.5,
                                    min_lam_diff=0.,
                                    combine_operator="min"):
    # reward 2
    if combine_operator == "min":
        combine_operator = min
    elif combine_operator == "average" or combine_operator == "mean":
        combine_operator = np.mean
    
    rs = []
    for rd, rp in zip(r_diff_list, r_parsimony_list):
        lam = (1-min_lam_diff) * sigmoid(3 / (min(error_tolerance, 1-error_tolerance)) * (rd-error_tolerance))
        rs.append(
            (1-lam)*rd + lam*rp
        )
    return combine_operator(rs)

def combine_rewards_epsilon_sigmoid_before(r_diff_list, r_parsimony_list, 
                                    error_tolerance=0.5,
                                    min_lam_diff=0.,
                                    combine_operator="min"):
    if combine_operator == "min":
        combine_operator = min
    elif combine_operator == "average" or combine_operator == "mean":
        combine_operator = np.mean
    
    rs = []
    for rd, rp in zip(r_diff_list, r_parsimony_list):
        lam = (1-min_lam_diff) * sigmoid(3 / (error_tolerance/2) * (rd-error_tolerance/2))
        rs.append(
            (1-lam)*rd + lam*rp
        )
    return combine_operator(rs)


def score_with_est(eq, tree_size, data_list, eta,
                   combine_rewards=combine_rewards_original,
                   task_ode_num=1, t_limit = 100.0,
                   reward_rescale=False, error_tolerance=0.5,
                   combine_operator="min", loss_func="L2", data_t_series_list=None, variable_list=None, vf_func_num=50):
    """
    Calculate reward score for a complete parse tree 
    If placeholder C is in the equation, also excute estimation for C 
    Reward = 1 / (1 + MSE) * Penalty ** num_term 

    Parameters
    ----------
    eq : Str object.
        the discovered equation (with placeholders for coefficients). 
    tree_size : Int object.
        number of production rules in the complete parse tree. 
    data : 2-d numpy array.
        measurement data, including independent and dependent variables (last row). 
    t_limit : Float object.
        time limit (seconds) for single evaluation, default 1 second. 
        
    Returns
    -------
    score: Float
        discovered equations. 
    eq: Str
        discovered equations with estimated numerical values. 
    """
    data_y_noise = data_list[0]
    data_dy_noise = data_list[1]

    full_terms, terms, coefficient_terms = extract(eq)
    if len(terms) >= 5 or tree_size >= 10:
        return 0, eq
    eq = math_enc(eq)
    assert loss_func in ["L2", "VF"]
    vf_criterion: VF_Loss = VF_Loss(func_num=vf_func_num, integ_method='simps')

    r_diff_list, r_parsimony_list, eqs = [], [], []
    if isinstance(eq, list):
        initial_eqs = eq
    else:
        initial_eqs = [eq] * len(data_y_noise)
        
    for env, (one_data_y_noise, one_data_dy_noise) in enumerate(zip(data_y_noise, data_dy_noise)):
        n_dynamic = len(one_data_y_noise)

        eq = initial_eqs[env]
        r_parsimony_sum = 0.0
        r_diff_sum = 0.0
        dynamic_count = 0

        ## define independent variables and dependent variable
        for i_dynamic in range(n_dynamic):
            for i_var, variable in enumerate(variable_list):
                globals()[variable] = one_data_y_noise[i_dynamic, :, i_var]
            target_variable = task_ode_num - 1
            origin_variable = task_ode_num - 1
            globals()['f_true'] = one_data_dy_noise[i_dynamic, :, target_variable]
            globals()['y_true'] = one_data_y_noise[i_dynamic, :, origin_variable]


            ## count number of numerical values in eq
            c_count = eq.count('C')
            with time_limit(t_limit, 'sleep'):
                try:
                    if c_count == 0:       ## no numerical values
                        f_pred = eval(eq)
                    elif c_count >= 10:    ## discourage over complicated numerical estimations
                        return 0, eq
                    else:                  ## with numerical values: coefficient estimation with Powell method
                        c_lst = ['c'+str(i) for i in range(c_count)]
                        for c in c_lst:
                            eq = eq.replace('C', c, 1)

                        def eq_test(c):
                            if "nan" in eq:
                                return float("inf")
                            for i in range(len(c)): globals()['c'+str(i)] = c[i]
                            eq_diff = np.linalg.norm(eval(eq) - f_true, 2)
                            return eq_diff

                        x0 = [1.0] * len(c_lst)
                        c_lst = minimize(eq_test, x0, method='Powell', tol=1e-6).x.tolist()
                        eq_est = eq
                        for i in range(len(c_lst)):
                            eq_est = eq_est.replace('c'+str(i), str(c_lst[i]), 1)
                        eq = eq_est.replace('+-', '-')
                        if "nan" in eq:
                            return 0, eq
                        f_pred = eval(eq)
                except Exception as e:
                    print(f"Error in evaluating {eq}", e)
                    return 0, eq

            if loss_func == "VF":
                if isinstance(f_pred, float):
                    return 0, eq
                xx = np.stack([y_true], axis=1)
                ff = np.stack([f_pred], axis=1)
                dis = vf_criterion(ff, xx, data_t_series_list[env])
                r_diff = float(1 / (1.0 + dis))
                assert isinstance(r_diff, float)
            else:  # L2
                diff = f_pred - f_true
                r_diff = float(1.0 / (1.0 + np.sqrt(np.linalg.norm(diff, 2) ** 2 / f_true.shape[0])))
                assert isinstance(r_diff, float)
            r_parsimony = eta ** tree_size

            r_parsimony_sum += r_parsimony
            r_diff_sum += r_diff
            dynamic_count += 1
        r_diff_list.append(r_diff_sum / dynamic_count)
        r_parsimony_list.append(r_parsimony_sum / dynamic_count)
        eqs.append(math_dec(eq))

    try:
        simplified_eqs = simplify_eqs(eqs)
    except Exception as e:
        print(e)
        print(f"Failure in simplify_eqs(eqs). eqs:", eqs)
        simplified_eqs = eqs
    return (
        combine_rewards(r_diff_list, r_parsimony_list,
                        error_tolerance=error_tolerance,
                        combine_operator=combine_operator,
                        num_samples=[len(one_data_y_noise) for one_data_y_noise in data_y_noise]),
        simplified_eqs
    )