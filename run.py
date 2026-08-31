import json
import os.path
import time
import pickle
import argparse
import logging
import queue

from functools import partial
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from invariant_physics.dataset import ODE_DICT, get_now_string, get_dataset, evaluate_trajectory_rmse, get_partial_mask, load_argparse, print_argparse, load_data, most_common, judge_expression_equal
from invariant_physics.spl import (SplBase, simplify_eqs,
                                   score_with_est,
                                   combine_rewards_original,
                                   combine_rewards_epsilon_piecewise,
                                   combine_rewards_epsilon_sigmoid,
                                   combine_rewards_epsilon_sigmoid_before)
from invariant_physics.spl.production_rule_utils import to_common_simplified_skeleton
from invariant_physics.spl import purify_strategy
from invariant_physics.dataset import extract, remove_constant, determine_most_frequent_terms, transform_sympy, set_eq_precision, calculate_parameter_rmse, simplify_and_replace_constants

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# class IgnoreEmptyMessagesFilter(logging.Filter):
#     def filter(self, record):
#         # Check if the message is not just whitespace
#         return record.getMessage().strip() != ''


def one_run_operation(
        ode,
        d,
        num_run,
        i_run,
        exp_rate,
        max_module_init,
        num_transplant,
        data_train,
        grammars,
        nt_nodes,
        max_len,
        num_aug,
        func_score,
        max_added_grammar_count,
        force,
        forced_nodes,
        t_series_list,
        main_path,
        output_dir,
        log_save_term_trace_path,
        timestring,
        tree_size_strategy,
        transplant_step,
        module_grow_step,
        data_val,
        log_truth_ode_list,
        variable_list,
        purification_threshold,
        seed,
        eta,
        task,
        task_ode_num,
):
    # try:
    np.random.seed(seed + i_run)
    # print("test", i_run)
    best_solution = ('Init', -1)

    exploration_rate = exp_rate
    max_module = max_module_init
    reward_his = []
    best_modules = []
    aug_grammars = []
    added_basic_grammars = []

    best_train_dic = dict()
    best_val_dic = dict()

    start_time = time.time()
    discovery_time = 0
    # full_data = np.zeros([num_env, len(train_samples[0]), len(d["curve_names"])])
    # for i_env, item in enumerate(train_samples):
    #     for j in range(len(d["curve_names"])):
    #         print(f"full data shape: {full_data.shape} j= {j} d[curve_names][1 + j]={d['curve_names'][1 + j]}")
    #         val = train_samples[i_env][d["curve_names"][1 + j]]
    #         full_data[i_env, :, j] = val

    num_env = ode.args.num_env

    for i_itr in range(num_transplant):

    # print(f"\nTransplant #{i_itr}")
    # print(f"ode.train_indices={ode.train_indices}")

    # save_path = os.path.join(main_path, output_dir, task, f"splbase_{num_env}_{eta}_{i_itr}_{i_test}_min.pkl")

    # if resume and os.path.exists(save_path):
    #     with open(save_path, "rb") as f:
    #         spl_model = pickle.load(f)
    #     print(f"Resuming model from iteration {spl_model.current_episode}")
    # else:

        spl_model = SplBase(data_sample=data_train,
                            base_grammars=grammars,
                            aug_grammars=aug_grammars,
                            added_basic_grammars=added_basic_grammars,
                            nt_nodes=nt_nodes,
                            max_len=max_len,
                            max_module=max_module,
                            aug_grammars_allowed=num_aug,
                            func_score=func_score,
                            exploration_rate=exploration_rate,
                            eta=eta,
                            max_added_grammar_count=max_added_grammar_count,
                            force=force,
                            forced_nodes=forced_nodes,
                            data_t_series_list=t_series_list,
                            i_transplant=0,
                            i_test=i_run,
                            output_dir=os.path.join(main_path, output_dir),
                            task=task,
                            num_env=num_env,
                            term_trace_path=log_save_term_trace_path,
                            variable_list=d["curve_names"],
                            timestring=timestring,
                            tree_size_strategy=tree_size_strategy,
                            )

        _, current_solution, good_modules = spl_model.run(transplant_step,
                                                          num_play=10,
                                                          print_flag=True)
        added_basic_grammars = spl_model.added_basic_grammars

        if not best_modules:
            best_modules = good_modules
        else:
            all_modules = best_modules + good_modules
            best_modules = sorted(list(
                [i for n, i in enumerate(all_modules) if i not in all_modules[:n]]
            ),
                key=lambda x: x[1])
        aug_grammars = [x[0] for x in best_modules[-num_aug:]]
        # print(f"aug_grammars: {aug_grammars}")

        # print([simplify_eq(x[2]) for x in best_modules[-num_aug:]])

        reward_his.append(best_solution[1])

        if current_solution[1] > best_solution[1]:
            best_solution = current_solution
        max_module += module_grow_step
        exploration_rate *= 5
        # print("best_modules[-1]: ", best_modules[-1])

        best_solution_print_train = simplify_eqs(best_solution[0])
        # print('best solution train: {}'.format(best_solution_print_train))
        best_train_dic[i_run] = best_solution_print_train
        # simplify_eqs(best_solution[0])
        # check if solution is discovered. Early stop if it is.

        eq_generated = spl_model.tree_to_eq(['f->A'] + best_modules[-1][0].split(','))
        # print(f"#### eq_generated: {eq_generated}")
        val_score, val_eqs = func_score(eq_generated, 0, data_val, eta=eta, data_t_series_list=t_series_list,
                                        variable_list=d["curve_names"])  # func_score
        # print(f"test scores: {test_score:.4f}")
        ##best_solution_print_test = eqs
        best_solution_print_val = simplify_eqs(val_eqs)
        # print('best solution val: {}'.format(best_solution_print_val))
        best_val_dic[i_run] = best_solution_print_val

        # except Exception as e:
        #     print(f"Error: test {i_test} failed:", e)

        # all_eqs.append(simplify_eqs(best_solution[0]))

        print(f'Run {i_run + 1:02d} / {num_run:02d} Iter {i_itr + 1:02d} / {num_transplant:02d}: train reward = {best_solution[1]:.6f}, train eqs = {best_solution_print_train}, val reward = {val_score:.6f}, val eqs = {best_solution_print_val}')
        # print('best solution: {}'.format(simplify_eqs(best_solution[0])))
        # print('best solution: {}'.format(best_solution[0]))
        # print('test score: {}'.format(test_score))
    print(f"[i_run = {i_run}] best_train_dic:\n", json.dumps(best_train_dic, indent=4))
    print(f"[i_run = {i_run}] best_val_dic:\n", json.dumps(best_val_dic, indent=4))

    # purifying

    one_run_val_dic = dict()
    one_run_val_dic["reward"] = val_score

    # tmp_group_raw, tmp_group_purified = 0, 0

    for i_env in range(num_env):
        truth_ode = str(log_truth_ode_list[i_env][task_ode_num - 1])
        predicted_ode = str(best_train_dic[i_run][i_env])
        _, truth_ode_terms, _ = extract(truth_ode)
        _, predicted_ode_terms, _ = extract(predicted_ode)
        # t_puri_0 = time.time()
        purify_res, purify_res_ratio, _, _ = purify_strategy(predicted_ode, data_val, variable_list,
                                                             threshold=purification_threshold)
        # print(f"purification method costs {time.time() - t_puri_0:.4f} s")
        purified_predicted_ode = str(purify_res)
        _, purified_predicted_ode_terms, _ = extract(purified_predicted_ode)

        # truth_ode_terms = remove_constant(truth_ode_terms)
        # predicted_ode_terms = remove_constant(predicted_ode_terms)
        # purified_predicted_ode_terms = remove_constant(purified_predicted_ode_terms)

        match_raw = (str(truth_ode_terms) == str(predicted_ode_terms)) or (
                simplify_and_replace_constants(truth_ode) == simplify_and_replace_constants(predicted_ode))
        # count_raw += int(match_raw)
        match_purified = (str(truth_ode_terms) == str(purified_predicted_ode_terms)) or (
                simplify_and_replace_constants(truth_ode) == simplify_and_replace_constants(
            purified_predicted_ode))
        # count_purified += int(match_purified)

        one_run_val_dic[i_env] = {
            "truth_ode": truth_ode,
            "truth_terms": str(truth_ode_terms),
            "predicted_ode": predicted_ode,
            "predicted_terms": str(predicted_ode_terms),
            "predicted_match": match_raw,
            "term_weight_distribution": str(purify_res_ratio).replace(", ", "/").replace("[", "").replace("]",
                                                                                                          ""),
            "purify_threshold": purification_threshold,
            "purified_predicted_ode": purified_predicted_ode,
            "purified_predicted_terms": str(purified_predicted_ode_terms),
            "purified_predicted_match": match_purified,
        }

        # tmp_group_raw += int(match_raw)
        # tmp_group_purified += int(match_purified)
        # if (i_env + 1) % num_env == 0:
        #     if tmp_group_raw >= num_env / 2:
        #         count_group_raw += 1
        #     tmp_group_raw = 0
        #
        #     if tmp_group_purified >= num_env / 2:
        #         count_group_purified += 1
        #     tmp_group_purified = 0
    print(f"run_val_dic: \n", json.dumps(one_run_val_dic, indent=4))
    # best_val_dic_collection[i_run * num_transplant + i_itr] = one_run_val_dic
    # best_val_dic_collection_reward_list.append([one_run_val_dic[0]["purified_predicted_terms"], val_score])
    purified_predicted_terms_list = [one_run_val_dic[i_env]["purified_predicted_terms"] for i_env in range(num_env)]
    predicted_terms_list = [one_run_val_dic[i_env]["predicted_terms"] for i_env in range(num_env)]
    # return one_run_val_dic, [one_run_val_dic[0]["purified_predicted_terms"], val_score], [one_run_val_dic[0]["predicted_terms"], val_score]
    return one_run_val_dic, [most_common(purified_predicted_terms_list), val_score], [most_common(predicted_terms_list), val_score]


def run_one(args):
    # Unpacking the arguments from the tuple
    i_run, ode, d, num_run, exp_rate, max_module_init, num_transplant, data_train, grammars, nt_nodes, max_len, num_aug, func_score, max_added_grammar_count, force, forced_nodes, t_series_list, main_path, output_dir, log_save_term_trace_path, timestring, tree_size_strategy, transplant_step, module_grow_step, data_val, log_truth_ode_list, variable_list, purification_threshold, seed, eta, task, task_ode_num = args

    # Call one_run_operation with unpacked arguments
    return one_run_operation(
        ode,
        d,
        num_run,
        i_run,
        exp_rate,
        max_module_init,
        num_transplant,
        data_train,
        grammars,
        nt_nodes,
        max_len,
        num_aug,
        func_score,
        max_added_grammar_count,
        force,
        forced_nodes,
        t_series_list,
        main_path,
        output_dir,
        log_save_term_trace_path,
        timestring,
        tree_size_strategy,
        transplant_step,
        module_grow_step,
        data_val,
        log_truth_ode_list,
        variable_list,
        purification_threshold,
        seed,
        eta,
        task,
        task_ode_num,
    )


def parallel_run(num_run, ode, d, exp_rate, max_module_init, num_transplant, data_train, grammars, nt_nodes, max_len,
                 num_aug, func_score, max_added_grammar_count, force, forced_nodes, t_series_list, main_path,
                 output_dir, log_save_term_trace_path, timestring, tree_size_strategy, transplant_step,
                 module_grow_step, data_val, log_truth_ode_list, variable_list, purification_threshold, seed, eta, task, task_ode_num):
    best_val_dic_collection = []
    best_val_dic_collection_reward_list = []
    best_val_dic_collection_reward_non_purified_list = []

    # Prepare argument tuples for each run (packing them as tuples)
    parallel_run_args = [
        (
            i_run, ode, d, num_run, exp_rate, max_module_init, num_transplant, data_train, grammars, nt_nodes, max_len,
            num_aug, func_score, max_added_grammar_count, force, forced_nodes, t_series_list, main_path, output_dir,
            log_save_term_trace_path, timestring, tree_size_strategy, transplant_step, module_grow_step, data_val,
            log_truth_ode_list, variable_list, purification_threshold, seed, eta, task, task_ode_num
        ) for i_run in range(num_run)
    ]

    # Use ProcessPoolExecutor to parallelize the task
    with ProcessPoolExecutor() as executor:
        # Pass the prepared args to run_one using executor.map
        results = list(executor.map(run_one, parallel_run_args))

    # Collect results after parallel processing
    for one_best_val_dic_collection, one_best_val_dic_collection_reward, one_best_val_dic_collection_reward_non_purified in results:
        best_val_dic_collection.append(one_best_val_dic_collection)
        best_val_dic_collection_reward_list.append(one_best_val_dic_collection_reward)
        best_val_dic_collection_reward_non_purified_list.append(one_best_val_dic_collection_reward_non_purified)

    return best_val_dic_collection, best_val_dic_collection_reward_list, best_val_dic_collection_reward_non_purified_list

def run_spl(
    task,
    task_ode_num,
    num_env, num_run, transplant_step, eta,
    data_dir='data/', max_len = 50,
    max_module_init = 10, num_aug = 5, exp_rate = 1/np.sqrt(2), num_transplant = 20,
    norm_threshold=1e-5, count_success = True, output_dir='results', max_added_grammar_count=4, force=True,
    forced_nodes=[], use_new_reward=1, reward_rescale=False, error_tolerance=0.99, loss_func="L2",
    combine_operator="min",
    min_lam_diff=0.,
    resume=True,
    timestring="",):
    """
    Executes the main training loop of Symbolic Physics Learner.
    
    Parameters
    ----------
    task : String object.
        benchmark task name. 
    num_env : Int object.
        number of environments in the problem
    num_run : Int object.
        number of runs performed.
    transplant_step : Int object.
        number of iterations simulated for training between two transplantations. 
    data_dir : String object.
        directory of training data samples. 
    max_len : Int object.
        maximum allowed length (number of production rules ) of discovered equations.
    eta : Int object.
        penalty factor for rewarding. 
    max_module_init : Int object.
        initial maximum length for module transplantation candidates. 
    num_aug : Int object.
        number of trees for module transplantation. 
    exp_rate : Int object.
        initial exploration rate. 
    num_transplant : Int object.
        number of transplantation candidate update performed throughout traning. 
    norm_threshold : Float object.
        numerical error tolerance for norm calculation, a very small value. 
    count_success : Boolean object. 
        if success rate is recorded. 
        
    Returns
    -------
    all_eqs: List<Str>
        discovered equations. 
    success_rate: Float
        success rate of all runs performed. 
    all_times: List<Float>
        runtimes for successful runs. 
    """
    
    ## define production s and non-terminal nodes.
    # log_start_time = get_now_string()
    log_start_time = timestring
    ode = get_dataset(timestring)
    if not ode.args.load_data_from_existing:
        ode.build()
    d = ode.params_config  # d = ODE_DICT[task].params_config
    grammars = d["rule_map"]
    nt_nodes = d["ntn_list"]
    ode_dim = d["ode_dim"]
    purification_threshold = d["purification_threshold"]
    variable_list = d["curve_names"]

    noise_ratio = ode.args.noise_ratio
    seed = ode.args.seed
    eta = ode.args.eta
    combine_operator = ode.args.combine_operator
    dataset_sparse = ode.args.dataset_sparse
    main_path = ode.args.main_path
    n_data_samples = ode.args.n_data_samples
    tree_size_strategy = ode.args.tree_size_strategy
    n_partial = ode.args.n_partial
    non_ode_sampling = ode.args.non_ode_sampling
    n_dynamic_string = str(ode.args.n_dynamic)
    n_dynamic_list_string = str(ode.args.n_dynamic_list).replace(", ", "/").replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace(",", "")
    # print(f"grammars={grammars}")
    # print(f"nt_nodes={nt_nodes}")
    with open(f'{main_path}/{data_dir}/{task}/{log_start_time}/csv/{task}_info.json', 'r') as f:
        info = json.load(f)
    log_truth_ode_list = []
    for i in range(num_env):
        log_truth_ode = [item.format(*info["params"][str(i)]) for item in (d['truth_ode_format'] if d.get('truth_ode_format') else [])]
        log_truth_ode_list.append(log_truth_ode)
    truth_eq_from_config = d['truth_ode_format'][task_ode_num - 1].format(*[1.0 for _ in range(len(d["random_params_base"]))])
    truth_eq_from_config = simplify_and_replace_constants(truth_eq_from_config)


    log_save_folder = f"{main_path}/logs/{task}/"
    log_summary_save_folder = f"{main_path}/logs/summary/"
    log_save_detail_path = f"{main_path}/logs/{task}/{log_start_time}.csv"
    log_save_term_trace_path = f"{main_path}/logs/{task}/{log_start_time}_term_trace.png"
    log_save_train_dic_full_path = f"{main_path}/logs/{task}/{log_start_time}_full_train.pkl"
    log_save_run_list_before_puri_full_path = f"{main_path}/logs/{task}/{log_start_time}_before_puri.pkl"
    log_save_run_list_after_puri_full_path = f"{main_path}/logs/{task}/{log_start_time}_after_puri.pkl"
    # log_path = f"logs/{task}/logs_{task}.csv"
    log_path = f"{log_summary_save_folder}/logs_{task}.csv"
    log_path_begin = f"{log_summary_save_folder}/logs_{task}_begin.csv"
    log_path_end = f"{log_summary_save_folder}/logs_{task}_end.csv"
    log_path_results = f"{log_summary_save_folder}/logs_{task}_results.csv"

    try:
        if not os.path.exists(log_save_folder):
            os.makedirs(log_save_folder)
    except Exception as e:
        print(e)
    try:
        if not os.path.exists(log_summary_save_folder):
            os.makedirs(log_summary_save_folder)
    except Exception as e:
        print(e)
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s,%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # logger = logging.getLogger()
    # logger.addFilter(IgnoreEmptyMessagesFilter())
    if not os.path.exists(log_path_end):
        with open(log_path_end, "w") as f:
            f.write(f"start_time,status,end_time,task,num_run,num_env,n_data_samples,avg_mse,avg_rmse,avg_relative_mse,avg_relative_rmse,purification_threshold,term_success_predicted_rate,term_success_predicted_purified_rate,reward_func_id,loss_func,eta,noise_ratio,task_ode_num,dataset_sparse,n_dynamic,n_dynamic_list,combine_operator,param_rmse,non_ode_sampling,tree_size_strategy,n_partial,truth_eq,most_frequent_eq_non_purified,most_frequent_eq,n_dynamic,vf_func_num,time_cost,seed\n")
    if not os.path.exists(log_path_begin):
        with open(log_path_begin, "w") as f:
            f.write(
                f"start_time,status,end_time,task,num_run,num_env,n_data_samples,avg_mse,avg_rmse,avg_relative_mse,avg_relative_rmse,purification_threshold,term_success_predicted_rate,term_success_predicted_purified_rate,reward_func_id,loss_func,eta,noise_ratio,task_ode_num,dataset_sparse,n_dynamic,n_dynamic_list,combine_operator,param_rmse,non_ode_sampling,tree_size_strategy,n_partial,truth_eq,most_frequent_eq_non_purified,most_frequent_eq,n_dynamic,vf_func_num,time_cost,seed\n")
    if not os.path.exists(log_path):
        logging.info(f"start_time,status,end_time,task,num_run,num_env,n_data_samples,avg_mse,avg_rmse,avg_relative_mse,avg_relative_rmse,purification_threshold,term_success_predicted_rate,term_success_predicted_purified_rate,reward_func_id,loss_func,eta,noise_ratio,task_ode_num,dataset_sparse,n_dynamic,n_dynamic_list,combine_operator,param_rmse,non_ode_sampling,tree_size_strategy,n_partial,truth_eq,most_frequent_eq_non_purified,most_frequent_eq,n_dynamic,vf_func_num,time_cost,seed")

    with open(log_path_begin, "a") as f:
        f.write(f"{log_start_time},Begin,{None},{task},{num_run},{num_env},{n_data_samples},{None},{None},{None},{None},{purification_threshold},{None},{None},{use_new_reward},{loss_func},{eta},{noise_ratio:.6f},{task_ode_num},{dataset_sparse},{n_dynamic_string},{n_dynamic_list_string},{combine_operator},{-1.0},{non_ode_sampling},{tree_size_strategy},{n_partial},{truth_eq_from_config},{None},{None},{ode.args.n_dynamic},{ode.args.vf_func_num},{None},{seed}\n")

    # Data loader: Updated
    data_load_path = os.path.join(main_path, data_dir, task, log_start_time, "dump", "data.pkl")
    _, data_train, data_val, data_test = load_data(data_load_path)


    # ## read training and testing data
    # train_samples = [] # shape: 5 * 10 * 500 * (*)
    #
    # for env in range(num_env):
    #     train_sample = [pd.read_csv(os.path.join(main_path, data_dir, task, log_start_time, f'{task}_train_{env}_{i_dynamic}.csv')) for i_dynamic in range(ode.n_dynamic)]
    #     # print(f"@@@@@ train sample: {train_sample[:20]}")
    #     # test_sample = pd.read_csv(os.path.join(main_path, data_dir, task, log_start_time, f'{task}_test_{env}.csv'))
    #     train_samples.append(train_sample)
    #     # test_samples.append(test_sample)
    #
    # # full_data_test = np.zeros([num_env, max(ode.num_test_list), ode_dim])
    # # for i in range(num_env):
    # #     full_data_path_test = os.path.join(main_path, data_dir, task, log_start_time, f'{task}_test_{i}.csv')
    # #     full_data_test[i, :ode.num_test_list[i], :] = np.asarray(
    # #         # pd.read_csv(os.path.join("data/", "Lotka_Volterra", f'Lotka_Volterra_test_{i}.csv'))
    # #         pd.read_csv(full_data_path_test)
    # #     )[:, 1: 1 + ode_dim]
    # if "t" in d["curve_names"]:
    #     dim_offset = 1
    # else:
    #     dim_offset = 0
    # full_data_train = [np.zeros([ode.n_dynamic, ode.num_train_list[i], ode_dim + dim_offset]) for i in range(num_env)]
    # for i in range(num_env):
    #     for i_dynamic in range(ode.n_dynamic):
    #         full_data_path_train = os.path.join(main_path, data_dir, task, log_start_time, f'{task}_train_{i}_{i_dynamic}.csv')
    #     # print(i, full_data_train[0].shape)
    #     # print(i, pd.read_csv(full_data_path_train).shape)
    #         if dim_offset:
    #             full_data_train[i][i_dynamic, :ode.num_train_list[i], :] = np.asarray(
    #                 # pd.read_csv(os.path.join("data/", "Lotka_Volterra", f'Lotka_Volterra_train_{i}.csv'))
    #                 pd.read_csv(full_data_path_train)
    #             )[:ode.num_train_list[i], 0: 1 + ode_dim]
    #         else:
    #             full_data_train[i][i_dynamic, :ode.num_train_list[i], :] = np.asarray(
    #                 # pd.read_csv(os.path.join("data/", "Lotka_Volterra", f'Lotka_Volterra_train_{i}.csv'))
    #                 pd.read_csv(full_data_path_train)
    #             )[:ode.num_train_list[i], 1: 1 + ode_dim]

    
    # num_success = 0
    # all_times = []
    # all_eqs = []
    
    ## number of module max size increase after each transplantation
    module_grow_step = (max_len - max_module_init) / num_transplant
    
    func_score = score_with_est   
    if use_new_reward == 0:
        func_score = partial(func_score, 
                             combine_rewards=combine_rewards_original)
    elif use_new_reward == 1:
        func_score = partial(func_score, 
                             combine_rewards=combine_rewards_epsilon_piecewise)
    elif use_new_reward == 2:
        combine_func = partial(combine_rewards_epsilon_sigmoid,
                               min_lam_diff=min_lam_diff)
        func_score = partial(func_score, 
                             combine_rewards=combine_func,)
    elif use_new_reward == 3:
        combine_func = partial(combine_rewards_epsilon_sigmoid_before,
                               min_lam_diff=min_lam_diff)
        func_score = partial(func_score, 
                             combine_rewards=combine_func,)
    else:
        raise Exception(f"Reward {use_new_reward} not implemented")
        
    print(f"Loss Function: {loss_func}")
    func_score = partial(func_score, 
                         error_tolerance=error_tolerance,
                         combine_operator=combine_operator,
                         task_ode_num=task_ode_num, 
                         reward_rescale=reward_rescale, loss_func=loss_func, vf_func_num=ode.args.vf_func_num)



    f_log = open(log_save_detail_path, "a")
    f_log.write(f"{log_start_time},{task},d{d['curve_names'][task_ode_num - 1]}\n")
    f_log.write(f"test_id,env_id,truth_ode,predicted_ode(un-purified),match(un-purified),predicted_ode(purified),match(purified),purify_threshold,term_weight_distribution\n")
        # for i_test in range(num_run):
        #     for i_env in range(num_env):
        #         f.write(f"{i_test},{i_env},'{log_truth_ode_list[i_env][task_ode_num - 1]}','{best_train_dic[i_test][i_env]}','{best_test_dic[i_test][i_env]}'\n")

    count_group_raw, count_group_purified = 0, 0
    t_series_list = ode.t_series_list
    # data_t_series_list_train = [ode.t_series[one_train_indices] for one_train_indices in ode.train_indices_list]
    # data_t_series_list_test = [ode.t_series[one_test_indices] for one_test_indices in ode.test_indices_list]

    true_ode_list = [str(log_truth_ode_list[i_env][task_ode_num - 1]) for i_env in range(num_env)]

    time_cost_start = time.time()

    # sequential
    # best_val_dic_collection = []
    # best_val_dic_collection_reward_list = []
    #
    # for i_run in range(num_run):
    #     one_best_val_dic_collection, one_best_val_dic_collection_reward = one_run_operation(
    #         ode,
    #         d,
    #         num_run,
    #         i_run,
    #         exp_rate,
    #         max_module_init,
    #         num_transplant,
    #         data_train,
    #         grammars,
    #         nt_nodes,
    #         max_len,
    #         num_aug,
    #         func_score,
    #         max_added_grammar_count,
    #         force,
    #         forced_nodes,
    #         t_series_list,
    #         main_path,
    #         output_dir,
    #         log_save_term_trace_path,
    #         timestring,
    #         tree_size_strategy,
    #         transplant_step,
    #         module_grow_step,
    #         data_val,
    #         log_truth_ode_list,
    #         variable_list,
    #         purification_threshold,
    #         seed,
    #         eta,
    #         task,
    #         task_ode_num,
    #     )
    #     best_val_dic_collection.append(one_best_val_dic_collection)
    #     best_val_dic_collection_reward_list.append(one_best_val_dic_collection_reward)

    # parallel
    best_val_dic_collection, best_val_dic_collection_reward_list, best_val_dic_collection_reward_non_purified_list = parallel_run(
        num_run,
        ode,
        d,
        exp_rate,
        max_module_init,
        num_transplant,
        data_train,
        grammars,
        nt_nodes,
        max_len,
        num_aug,
        func_score,
        max_added_grammar_count,
        force,
        forced_nodes,
        t_series_list,
        main_path,
        output_dir,
        log_save_term_trace_path,
        timestring,
        tree_size_strategy,
        transplant_step,
        module_grow_step,
        data_val,
        log_truth_ode_list,
        variable_list,
        purification_threshold,
        seed,
        eta,
        task,
        task_ode_num,
    )

    time_cost_end = time.time()
    time_cost_period = time_cost_end - time_cost_start

    print(f"[{get_now_string('%Y-%m-%d %H:%M:%S.%f')}] time_cost_period: {time_cost_period:.6f} s")

    # term_success_predicted_rate = count_group_raw / num_run
    # term_success_predicted_msg = f"term_success_predicted_rate: {count_group_raw} / {num_run} = {term_success_predicted_rate:.6f}"
    # term_success_predicted_purified_rate = count_group_purified / num_run
    # term_success_predicted_purified_msg = f"term_success_predicted_purified_rate: {count_group_purified} / {num_run} = {term_success_predicted_purified_rate:.6f}"

    # f_log.write(f"noise_ratio: {noise_ratio:.2f}" + "\n")
    # f_log.write(f"seed: {seed}" + "\n")
    # f_log.write(term_success_predicted_msg + "\n")
    # f_log.write(term_success_predicted_purified_msg + "\n")
    # print(term_success_predicted_msg)
    # print(term_success_predicted_purified_msg)
    
    # success_rate = num_success / num_run
    # if count_success:
    #     print('success rate :', success_rate)
    # print(f"[{get_now_string('%Y-%m-%d %H:%M:%S.%f')}] best_train_dic:")
    # print(json.dumps(best_train_dic, indent=4))
    # print("best_test_dic:")
    # print(json.dumps(best_test_dic, indent=4))
    # print("best_test_dic_full (TEST ONLY):")
    # print(json.dumps(best_train_dic_full, indent=4))
    with open(log_save_train_dic_full_path, "wb") as f_full:
        pickle.dump(best_val_dic_collection, f_full)
    # print(f"best_test_dic_full (TEST ONLY) has been saved to '{log_save_train_dic_full_path}' successfully")
    # obtain the most frequent / highest-rewarded terms result
    if ode.args.select_run_strategy == "highest_reward":
        best_val_dic_collection_reward_list = sorted(best_val_dic_collection_reward_list, key=lambda x: -x[1])
        best_val_dic_collection_reward_non_purified_list = sorted(best_val_dic_collection_reward_non_purified_list, key=lambda x: -x[1])
        best_ode_terms = best_val_dic_collection_reward_list[0][0].replace("[", "").replace("]", "").split(", ")
        best_ode_terms_non_purified = best_val_dic_collection_reward_non_purified_list[0][0].replace("[", "").replace("]", "").split(", ")
        print(f"[{get_now_string('%Y-%m-%d %H:%M:%S.%f')}] best_val_dic_collection_reward_list: {best_val_dic_collection_reward_list}")
        print(f"[{get_now_string('%Y-%m-%d %H:%M:%S.%f')}] best_val_dic_collection_reward_non_purified_list: {best_val_dic_collection_reward_non_purified_list}")
    else:
        # best_ode_terms = determine_most_frequent_terms(best_val_dic_collection).replace("[", "").replace("]", "").split(", ")
        raise NotImplementedError
    print(f"[{get_now_string('%Y-%m-%d %H:%M:%S.%f')}] best_ode_terms: {best_ode_terms}")
    print(f"[{get_now_string('%Y-%m-%d %H:%M:%S.%f')}] best_ode_terms_non_purified: {best_ode_terms_non_purified}")
    assert len(best_ode_terms) > 0 and len(best_ode_terms_non_purified) > 0
    most_frequent_eq = transform_sympy("+".join(best_ode_terms))
    most_frequent_eq_non_purified = transform_sympy("+".join(best_ode_terms_non_purified))

    most_frequent_eq = simplify_and_replace_constants(most_frequent_eq)
    most_frequent_eq_non_purified = simplify_and_replace_constants(most_frequent_eq_non_purified)

    term_success_predicted_rate = judge_expression_equal(most_frequent_eq_non_purified, truth_eq_from_config)
    term_success_predicted_purified_rate = judge_expression_equal(most_frequent_eq, truth_eq_from_config)

    finalized_prediction_reward, finalized_predicted_eqs = func_score(most_frequent_eq, 0, data_test, eta=eta, data_t_series_list=t_series_list, variable_list=d["curve_names"]) # ode.t_series[ode.test_indices]
    truth_from_config_reward, truth_from_config_eqs = func_score(truth_eq_from_config, 0, data_test, eta=eta,
                                                                      data_t_series_list=t_series_list, variable_list=d["curve_names"])  # ode.t_series[ode.test_indices]
    print(f"[{get_now_string('%Y-%m-%d %H:%M:%S.%f')}] timestring: {timestring}")
    print(f"[{get_now_string('%Y-%m-%d %H:%M:%S.%f')}] (prediction) most_frequent_eq: {most_frequent_eq} reward (set tree_size = 0): {finalized_prediction_reward}")
    print(f"[{get_now_string('%Y-%m-%d %H:%M:%S.%f')}] (ground-truth) truth_from_config_eq: {truth_eq_from_config} reward (set tree_size = 0): {truth_from_config_reward}")
    # print("finalized_predicted_eqs:", finalized_predicted_eqs)
    finalized_predicted_eqs = simplify_eqs(finalized_predicted_eqs)
    mse_list, rmse_list, relative_mse_list, relative_rmse_list = np.zeros(num_env), np.zeros(num_env), np.zeros(num_env), np.zeros(num_env)

    # for i_env in range(num_env):
    #     mse, rmse, relative_mse, relative_rmse = evaluate_trajectory_rmse(ode, finalized_predicted_eqs[i_env], i_env, task_ode_num)
    #     mse_list[i_env], rmse_list[i_env] = mse, rmse
    #     relative_mse_list[i_env], relative_rmse_list[i_env] = relative_mse, relative_rmse
    avg_mse = np.mean(mse_list)
    avg_rmse = np.mean(rmse_list)
    avg_relative_mse = np.mean(relative_mse_list)
    avg_relative_rmse = np.mean(relative_rmse_list)

    finalized_predicted_eqs = [set_eq_precision(item, 12) for item in finalized_predicted_eqs]

    print(f"[{get_now_string('%Y-%m-%d %H:%M:%S.%f')}] Ground-truth eqs: {true_ode_list}")
    print(f"[{get_now_string('%Y-%m-%d %H:%M:%S.%f')}] Finalized predicted eqs: {finalized_predicted_eqs}")

    try:
        param_rmse = calculate_parameter_rmse(true_ode_list, finalized_predicted_eqs)
    except Exception as e:
        param_rmse = -1.0
        print(e)

    truth_eqs = [log_truth_ode_list[idx][task_ode_num - 1] for idx in range(num_env)]
    truth_eqs = [set_eq_precision(item, 12) for item in truth_eqs]

    log_end_time = get_now_string()
    with open(log_path_end, "a") as f:
        f.write(f"{log_start_time},End,{log_end_time},{task},{num_run},{num_env},{n_data_samples},{avg_mse},{avg_rmse},{avg_relative_mse},{avg_relative_rmse},{purification_threshold},{term_success_predicted_rate},{term_success_predicted_purified_rate},{use_new_reward},{loss_func},{eta},{noise_ratio:.6f},{task_ode_num},{dataset_sparse},{n_dynamic_string},{n_dynamic_list_string},{combine_operator},{param_rmse},{non_ode_sampling},{tree_size_strategy},{n_partial},{truth_eq_from_config},{most_frequent_eq_non_purified},{most_frequent_eq},{ode.args.n_dynamic},{ode.args.vf_func_num},{time_cost_period},{seed}\n")
    print(f"{log_start_time},End,{log_end_time},{task},{num_run},{num_env},{n_data_samples},{avg_mse},{avg_rmse},{avg_relative_mse},{avg_relative_rmse},{purification_threshold},{term_success_predicted_rate},{term_success_predicted_purified_rate},{use_new_reward},{loss_func},{eta},{noise_ratio:.6f},{task_ode_num},{dataset_sparse},{n_dynamic_string},{n_dynamic_list_string},{combine_operator},{param_rmse},{non_ode_sampling},{tree_size_strategy},{n_partial},{truth_eq_from_config},{most_frequent_eq_non_purified},{most_frequent_eq},{ode.args.n_dynamic},{ode.args.vf_func_num},{time_cost_period},{seed}\n")
    with open(log_path_results, "a") as f:
        f.write(f"{log_start_time},truth,{','.join(truth_eqs)}\n")
        f.write(f"{log_start_time},prediction,{','.join(finalized_predicted_eqs)}\n")
    logging.info(f"{log_start_time},End,{log_end_time},{task},{num_run},{num_env},{n_data_samples},{avg_mse},{avg_rmse},{avg_relative_mse},{avg_relative_rmse},{purification_threshold},{term_success_predicted_rate},{term_success_predicted_purified_rate},{use_new_reward},{loss_func},{eta},{noise_ratio:.6f},{task_ode_num},{dataset_sparse},{n_dynamic_string},{n_dynamic_list_string},{combine_operator},{param_rmse},{non_ode_sampling},{tree_size_strategy},{n_partial},{truth_eq_from_config},{most_frequent_eq_non_purified},{most_frequent_eq},{ode.args.n_dynamic},{ode.args.vf_func_num},{time_cost_period},{seed}")

    f_log.write(
        f"{log_start_time},End,{log_end_time},{task},{num_run},{num_env},{n_data_samples},{avg_mse},{avg_rmse},{avg_relative_mse},{avg_relative_rmse},{purification_threshold},{term_success_predicted_rate},{term_success_predicted_purified_rate},{use_new_reward},{loss_func},{eta},{noise_ratio:.6f},{task_ode_num},{dataset_sparse},{n_dynamic_string},{n_dynamic_list_string},{combine_operator},{param_rmse},{non_ode_sampling},{tree_size_strategy},{n_partial},{truth_eq_from_config},{most_frequent_eq_non_purified},{most_frequent_eq},{ode.args.n_dynamic},{ode.args.vf_func_num},{time_cost_period},{seed}\n")
    f_log.close()

    return

def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    args, parser = load_argparse()
    print(f"timestring: {args.timestring}")
    print(f"main_path: {args.main_path}")

    task = args.task
    task_ode_num = args.task_ode_num
    eta = args.eta
    num_env = args.num_env
    np.random.seed(args.seed)
    print("="*30)
    print(f"Task: {task} #{task_ode_num}\n")

    print("="*30)
    run_spl(
        task,
        task_ode_num,
        num_env=num_env,
        num_run=args.num_run,
        max_len=50,
        eta=eta,
        max_module_init=20,
        num_transplant=args.num_transplant,  # 1
        num_aug=5,
        transplant_step=args.transplant_step,
        count_success=True,
        output_dir=args.output_dir,
        max_added_grammar_count=args.max_added_grammar_count,
        force=args.force,
        # forced_nodes=[['A->(A+A)', 'A->(A*A)', 'A->(A*C)', 'A->x', 'A->y', 'A->(A*C)', 'A->x']],
        use_new_reward=args.use_new_reward,
        norm_threshold=0.01,
        data_dir=args.data_dir,
        reward_rescale=args.reward_rescale,
        error_tolerance=args.error_tolerance,
        loss_func=args.loss_func,
        combine_operator=args.combine_operator,
        min_lam_diff=args.min_lam_diff,
        resume=args.resume,
        timestring=args.timestring,
       )




