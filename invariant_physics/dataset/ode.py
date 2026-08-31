import os
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.integrate import odeint, solve_ivp
from tqdm import tqdm
import itertools

from ._utils import sample_lhs, save_to_csv, params_random, generate_ordered_indices, get_now_string
# from .gp import GPPCA0


class ODEDataset:
    def __init__(self, args, params_config, non_ode_function=False):
        self.args = args
        self.setup_seed(self.args.seed)
        self.params_config = params_config
        # self.n_data_samples_list = self.args.n_data_samples_list
        self.n_dynamic_list = self.args.n_dynamic_list
        # print(f"self.n_data_samples_list: {self.n_data_samples_list}")
        self.ode_dim = self.params_config["ode_dim"]
        self.ode_dim_function = self.params_config["ode_dim_function"]
        self.ode_name = self.params_config["task"]

        self.params, self.y0_list = self._get_ode_params_and_y0(self.args.num_env, self.args.params_strategy)
        print("self.params.shape:", [item.shape for item in self.params])
        print("self.y0_list.shape:", [item.shape for item in self.y0_list])
        self.params = self._set_partial_param(self.params, self.args.partial_mask_list)  # For partial masked odes
        # self.params_eval = [0.0 for i in range(len(self.params_config["curve_names"]))]
        self.t_series_list = []
        # self.train_indices_list, self.test_indices_list = [], []
        self.num_train_list, self.num_val_list, self.num_test_list = [], [], []
        self.train_index_list, self.val_index_list, self.test_index_list = [], [], []

        # self.num_train, self.num_test = None, None
        # self.n = int(self.params_config["t_max"] / self.params_config["dt"])
        self.dt = self.params_config["dt"]
        if not self.args.n_data_samples:
            self.N = int((self.params_config["t_max"] - self.params_config["t_min"]) / self.params_config["dt"])
            self.args.n_data_samples = self.N
        else:
            self.N = self.args.n_data_samples
        self._set_t()

        self.y = [np.zeros([self.n_dynamic_list[i], self.N, self.ode_dim]) for i in range(self.args.num_env)]  # y shape: 5*10*500*2 for LV model
        self.y_noise = [np.zeros([self.n_dynamic_list[i], self.N, self.ode_dim]) for i in range(self.args.num_env)]
        self.dy_noise = [np.zeros([self.n_dynamic_list[i], self.N, self.ode_dim_function]) for i in range(self.args.num_env)]

        # self.y = [np.zeros([self.n_dynamic, self.n, self.ode_dim]) for i in
        #           range(self.args.num_env)]  # y shape: 5*10*500*2 for LV model
        # self.y_noise = [np.zeros([self.n_dynamic, self.n, self.ode_dim]) for i in
        #                 range(self.args.num_env)]
        # self.dy_noise = [np.zeros([self.n_dynamic, self.n, self.ode_dim_function]) for i in
        #                  range(self.args.num_env)]

        self.non_ode_function = non_ode_function  # It's a direct function, not an ODE

        # if self.args.extract_csv:
        #     self._extract_csv()

        # self._build()

    def _set_partial_param(self, params, mask):
        assert len(params) == len(mask) == self.args.num_env, f"len(params)={len(params)}, len(mask)={len(mask)}, self.args.num_env={self.args.num_env} should be the same!"
        # For n_partial
        if self.ode_name == "Damped_Pendulum":
            param_idx = 1
            for i in range(self.args.num_env):
                params[i][param_idx] *= mask[i]
        return params


    def _get_ode_params_and_y0(self, num_env: int, params_strategy: str):
        assert num_env <= self.params_config.get("env_max")
        assert params_strategy in self.params_config.get("params_strategy_list")
        params_func = self.params_config["params"][params_strategy]
        params = params_func(
            num_env=num_env,
            default_list=self.params_config["default_params_list"],
            base=self.params_config["random_params_base"],
            seed=self.args.seed,
            random_rate=0.1,)
        y0_list = []
        for i_env in range(num_env):
            one_env_y0_list = [params_random(
                num_env=num_env,
                default_list=self.params_config["default_y0_list"],
                base=self.params_config["random_y0_base"],
                seed=self.args.seed,
                random_rate=0.8,
                seed_offset=i_dynamic,) for i_dynamic in range(self.args.n_dynamic_list[i_env])]
            y0_list.append(np.asarray(one_env_y0_list))
        return np.asarray(params), y0_list

    def _func(self, x, t, env_id):
        raise NotImplemented

    def _func_solve_ivp(self, t, x, env_id):
        return self._func(x, t, env_id)

    def _set_non_ode_y(self):
        raise NotImplemented

    def _set_t(self):
        assert self.args.sample_strategy in ["uniform", "lhs"]
        for i in range(self.args.num_env):
            if self.args.sample_strategy == "uniform":
                self.t_series_list.append(np.asarray([self.params_config["t_min"] + self.dt * j for j in range(self.N)])) ## classic
            else:  # lhs
                self.t_series_list.append(sample_lhs(self.params_config["t_min"], self.params_config["t_max"], self.N))


    def build(self):
        # train_test_total = self.args.train_test_total
        # self.num_train_list = [int(self.args.train_ratio * one_train_test_total) for one_train_test_total in self.args.train_test_total_list]
        # self.num_test = train_test_total - self.num_train
        # self.num_test_list = [one_train_test_total - one_num_train for one_train_test_total, one_num_train in zip(self.args.train_test_total_list, self.num_train_list)]
        # self.num_test_list = [int(self.args.test_ratio * one_train_test_total) for one_train_test_total in self.args.train_test_total_list]
        self.num_train_list = [int(one_n_dynamic * self.args.train_ratio) for one_n_dynamic in self.args.n_dynamic_list]
        self.num_val_list = [int(one_n_dynamic * self.args.val_ratio) for one_n_dynamic in self.args.n_dynamic_list]
        self.num_test_list = [int(one_n_dynamic * self.args.test_ratio) for one_n_dynamic in self.args.n_dynamic_list]
        for i_env in range(self.args.num_env):
            one_train_index, one_val_index, one_test_index = generate_ordered_indices(
                self.args.n_dynamic_list[i_env],
                self.num_train_list[i_env],
                self.num_val_list[i_env],
                self.num_test_list[i_env])

            self.train_index_list.append(one_train_index)
            self.val_index_list.append(one_val_index)
            self.test_index_list.append(one_test_index)

        if self.args.load_data_from_existing:
            return
        # indices = np.arange(self.n)
        # # for one_num_train, one_num_test in zip(self.num_train_list, self.num_test_list):
        # #     assert one_num_train + one_num_test <= self.n

        # if self.args.train_sample_strategy == "uniform":
        #     assert self.args.dataset_sparse in ["sparse", "dense"]
        #     train_indices_list, test_indices_list = [], []
        #     for i in range(self.args.num_env):
        #         one_num_train, one_num_test = self.num_train_list[i], self.num_test_list[i]
        #         if self.args.dataset_sparse == "sparse":
        #             train_indices_list.append(indices[0::(int(self.n / 1.01) // one_num_train)][:one_num_train])
        #             if one_num_test > 0:
        #                 test_indices_list.append(indices[0::(int(self.n / 1.01) // one_num_test)][:one_num_test])
        #         else:
        #             sample_freq = int(0.03 / self.dt) if int(0.05 / self.dt) >= 1 else 1
        #             train_indices_list.append(indices[::sample_freq][:one_num_train])
        #             if one_num_test > 0:
        #                 test_indices_list.append(indices[::sample_freq][:one_num_test])
        # else:  # random
        #     train_indices_list, test_indices_list = [], []
        #     for i in range(self.args.num_env):
        #         one_num_train, one_num_test = self.num_train_list[i], self.num_test_list[i]
        #         np.random.shuffle(indices)
        #         train_indices_list.append(sorted(indices[:one_num_train]))
        #         if one_num_test > 0:
        #             test_indices_list.append(sorted(indices[one_num_train: one_num_train + one_num_test]))
        # self.train_indices_list = train_indices_list
        # self.test_indices_list = test_indices_list
        # print(f"self.train_indices_list: length={[len(item) for item in self.train_indices_list]},", self.train_indices_list)
        # print(f"self.test_indices_list: length={[len(item) for item in self.test_indices_list]},", self.test_indices_list)

        save_folder = os.path.join(self.args.main_path, self.args.data_dir, self.ode_name, self.args.timestring, "csv")
        save_folder_dump = os.path.join(self.args.main_path, self.args.data_dir, self.ode_name, self.args.timestring, "dump")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if not os.path.exists(save_folder_dump):
            os.makedirs(save_folder_dump)

        print(f"ode_name: {self.ode_name}")
        print(f"num_env: {self.args.num_env}")
        print(f"t_min: {self.params_config['t_min']}, t_max: {self.params_config['t_max']}, dt: {self.params_config['dt']}, N: {self.N}")
        # print(f"train_test_total_list: {self.args.train_test_total_list}")

        print(f"train set: {self.num_train_list} ({self.args.train_ratio * 100:.1f} %), val set: {self.num_val_list} ({self.args.val_ratio * 100:.1f} %), test set: {self.num_test_list} ({self.args.test_ratio * 100:.1f} %), total: {self.args.n_dynamic}, N: {self.N}")
        # print(f"train set: {self.num_train} ({self.num_train / train_test_total * 100:.1f} %), test set: {self.num_test} ({self.num_test / train_test_total * 100:.1f} %), train_test_total: {train_test_total}, n_total: {self.n}")
        print(f"noise ratio: {self.args.noise_ratio}")
        print(f"save_folder: {save_folder}")

        self._save_task_info(save_folder)

        for i in range(self.args.num_env):
            print(f"Environment {i:02d}: params={[f'{item:.8f}' for item in self.params[i]]}, truth = {[item.format(*self.params[i]) for item in (self.params_config['truth_ode_format'] if self.params_config.get('truth_ode_format') else [])]}")

        for i in tqdm(range(self.args.num_env)):
            if self.args.env_id != None and i != self.args.env_id:
                continue
            # print(f"integrate_method: {self.args.integrate_method}")
            assert self.args.integrate_method in ["ode_int", "solve_ivp"]
            if not self.non_ode_function:
                for i_dynamic in range(self.n_dynamic_list[i]):
                    if self.args.integrate_method == "ode_int":
                        self.y[i][i_dynamic] = odeint(self._func, self.y0_list[i][i_dynamic], self.t_series_list[i], (i,))
                        # self.y[i][i_dynamic] = odeint(self._func, self.y0_list[i_dynamic][i], self.t_series_list[i], (i,))  ## classic
                    else:
                        sol = solve_ivp(fun=self._func_solve_ivp, t_span=(self.t_series_list[i][0], self.t_series_list[i][-1]), y0=np.asarray(self.y0_list[i][i_dynamic]), args=(i,), t_eval=self.t_series_list[i], method="RK45")
                        self.y[i][i_dynamic] = np.transpose(sol.y, (1, 0))
            else:
                assert NotImplementedError

            # self.y_noise[i] = self.y[i] * (1 + 2.0 * self.args.noise_ratio * (-0.5 + np.random.random(self.y[i].shape)))
            # print("self.y_noise[i].shape", self.y_noise[i].shape)
            for i_dynamic in range(self.n_dynamic_list[i]):
                std_base = np.std(self.y[i][i_dynamic], axis=0)
                noise_sigma = std_base * self.args.noise_ratio
                # noise_sigma = np.broadcast_to(noise_sigma, (2000, 2))
                # print("std_base.shape", std_base.shape)
                # print("noise shape: ", (noise_sigma * np.random.randn(*self.y_noise[i].shape)).shape)
                self.y_noise[i][i_dynamic] = self.y[i][i_dynamic] + noise_sigma * np.random.randn(*self.y[i][i_dynamic].shape)

                y_noise_train = self.y_noise[i][i_dynamic]
                y_noise_test = self.y_noise[i][i_dynamic]
                raw_path = os.path.join(save_folder, f"{self.ode_name}_{i}_{i_dynamic}_raw.csv")
                # save_to_csv(raw_path, [self.t_series_list[i]] + [self.y_noise[i][i_dynamic, :, j] for j in range(self.ode_dim)], ["t"] + self.params_config["curve_names"][:self.ode_dim])

                if self.args.save_figure:
                    t_train, t_test = self.t_series_list[i], self.t_series_list[i]
                    y_train, y_test = self.y[i][i_dynamic], self.y[i][i_dynamic]

                    if i_dynamic < 5:  # only plot the first 5 (at most) trajectories
                        save_figure_path = os.path.join(save_folder, f"{self.ode_name}_{i}_{i_dynamic}.png")
                        self._plot_dataset(save_figure_path, self.t_series_list[i], t_train, t_test, self.y[i][i_dynamic], y_train, y_test, y_noise_train, y_noise_test)

                for j in range(self.ode_dim):
                    if not self.non_ode_function:
                        self.dy_noise[i][i_dynamic, :, j] = np.gradient(self.y_noise[i][i_dynamic, :, j], self.t_series_list[i])
                    else:
                        self.dy_noise[i][i_dynamic, :, :] = self._func(self.y_noise[i][i_dynamic, :, :], None, env_id=i)

                # dy_path_train = os.path.join(save_folder, f"{self.ode_name}_train_{i}_{i_dynamic}.csv")
                # dy_path_test = os.path.join(save_folder, f"{self.ode_name}_test_{i}_{i_dynamic}.csv")

                # Generate points based on Gaussian Process of observation
    #             if self.args.dataset_gp:
    #                 # Generate with frequency = `freq` points per second
    #                 freq = 50
    #                 t_max = max(self.t_series_list[i])
    #                 t_min = min(self.t_series_list[i])
    #                 num_points = int(freq * (t_max-t_min))
    #                 t_train = np.linspace(t_min, t_max, num=num_points)
    #                 y_train_generated = np.zeros([num_points, self.ode_dim])
    #                 dy_noise_train = np.zeros([num_points, self.ode_dim])
    #                 for j in range(self.ode_dim):
    #                     pca = GPPCA0(y_noise_train[:, j].reshape(-1,1), self.t_series_list[i], noise_sigma[j], sigma_out=std_base[j])
    #                     y_train_generated[:, j] = pca.get_predictive(new_sample=1, t_new=t_train).reshape(-1)
    #                     y_train_mean = pca.get_predictive_mean(t_new=t_train).reshape(-1)
    #                     y_train_std = np.sqrt(np.diag(pca.get_X_cov(t_new=t_train)))
    #                     if self.args.save_figure:
    #                         save_path = os.path.join(save_folder, f"{self.ode_name}_GP_{i}.png")
    #                         self._plot_GP(save_path, t_train, y_train_generated[:, j], y_train_mean, y_train_std)
    # #                     dy_noise_train[:, j] = np.gradient(y_train_generated[:, j], t_train)
    #                 y_noise_train = y_train_generated
    #             else:

    #             t_train = self.t_series_list[i]
    #             dy_noise_train = self.dy_noise[i][i_dynamic, :, :]
    #
    #             save_to_csv(dy_path_train,
    #                         [t_train] + [y_noise_train[:, j] for j in range(self.ode_dim)] + [dy_noise_train[:, j] for j in range(self.ode_dim_function)],
    #                         ["t"] + self.params_config["curve_names"][:self.ode_dim] + ['d'+name for name in self.params_config["curve_names"][:self.ode_dim_function]])

                # save_to_csv(dy_path_test,
                #             [self.t_series_list[i]] + [y_noise_test[:, j] for j in range(self.ode_dim)] + [self.dy_noise[i][i_dynamic, :, j] for j in range(self.ode_dim_function)],
                #             ["t"] + self.params_config["curve_names"][:self.ode_dim] + ['d'+name for name in self.params_config["curve_names"][:self.ode_dim_function]])

                # save_to_csv(dy_path_train,
                #             [self.t_series_list[i][self.train_indices_list[i]]] + [y_noise_train[self.train_indices_list[i], j] for j in range(self.ode_dim)] + [self.dy_noise[i][i_dynamic, self.train_indices_list[i], j] for j in range(self.ode_dim_function)],
                #             ["t"] + self.params_config["curve_names"][:self.ode_dim] + ['d'+name for name in self.params_config["curve_names"][:self.ode_dim_function]])
                # save_to_csv(dy_path_test,
                #             [self.t_series_list[i][self.test_indices_list[i]]] + [y_noise_test[self.test_indices_list[i], j] for j in range(self.ode_dim)] + [self.dy_noise[i][i_dynamic, self.test_indices_list[i], j] for j in range(self.ode_dim_function)],
                #             ["t"] + self.params_config["curve_names"][:self.ode_dim] + ['d'+name for name in self.params_config["curve_names"][:self.ode_dim_function]])



        # self.args = args
        #         self.setup_seed(self.args.seed)
        #         self.params_config = params_config
        #         # self.n_data_samples_list = self.args.n_data_samples_list
        #         self.n_dynamic_list = self.args.n_dynamic_list
        #         # print(f"self.n_data_samples_list: {self.n_data_samples_list}")
        #         self.ode_dim = self.params_config["ode_dim"]
        #         self.ode_dim_function = self.params_config["ode_dim_function"]
        #         self.ode_name = self.params_config["task"]
        #
        #         self.params, self.y0_list = self._get_ode_params_and_y0(self.args.num_env, self.args.params_strategy)
        #         print("self.params.shape:", [item.shape for item in self.params])
        #         print("self.y0_list.shape:", [item.shape for item in self.y0_list])
        #         self.params = self._set_partial_param(self.params, self.args.partial_mask_list)  # For partial masked odes
        #         # self.params_eval = [0.0 for i in range(len(self.params_config["curve_names"]))]
        #         self.t_series_list = []
        #         # self.train_indices_list, self.test_indices_list = [], []
        #         self.num_train_list, self.num_val_list, self.num_test_list = [], [], []
        #         self.train_index_list, self.val_index_list, self.test_index_list = [], [], []
        #
        #         # self.num_train, self.num_test = None, None
        #         # self.n = int(self.params_config["t_max"] / self.params_config["dt"])
        #         self.dt = self.params_config["dt"]
        #         self.N = int((self.params_config["t_max"] - self.params_config["t_min"]) / self.params_config["dt"])
        #         self._set_t()
        #
        #         self.y = [np.zeros([self.n_dynamic_list[i], self.N, self.ode_dim]) for i in range(self.args.num_env)]  # y shape: 5*10*500*2 for LV model
        #         self.y_noise = [np.zeros([self.n_dynamic_list[i], self.N, self.ode_dim]) for i in range(self.args.num_env)]
        #         self.dy_noise = [np.zeros([self.n_dynamic_list[i], self.N, self.ode_dim_function]) for i in range(self.args.num_env)]

        if self.args.extract_csv:
            return

        data_dump = dict()

        data_dump["args"] = self.args
        data_dump["t_series_list"] = self.t_series_list
        data_dump["params_config"] = self.params_config
        data_dump["params"] = self.params
        data_dump["params_shape"] = [item.shape for item in self.params]
        data_dump["n_dynamic_list"] = self.n_dynamic_list
        data_dump["N"] = self.N

        for data_type in ["train", "val", "test"]:
            index_list = eval(f"self.{data_type}_index_list")

            one_type_y0_list = [item[index_list[i_env]] for i_env, item in enumerate(self.y0_list)]
            one_type_y = [item[index_list[i_env]] for i_env, item in enumerate(self.y)]
            one_type_y_noise = [item[index_list[i_env]] for i_env, item in enumerate(self.y_noise)]
            one_type_dy_noise = [item[index_list[i_env]] for i_env, item in enumerate(self.dy_noise)]

            one_type_data_dump = {
                "data_type": data_type,
                "dynamic_index_list": index_list,

                "y0_list": one_type_y0_list,
                "y0_list_shape": [item.shape for item in one_type_y0_list],
                "y": one_type_y,
                "y_shape": [item.shape for item in one_type_y],
                "y_noise": one_type_y_noise,
                "y_noise_shape": [item.shape for item in one_type_y_noise],
                "dy_noise": one_type_dy_noise,
                "dy_noise_shape": [item.shape for item in one_type_dy_noise],
            }
            exec(f"data_dump['data_{data_type}'] = one_type_data_dump")
        # print(data_dump)
        with open(os.path.join(save_folder_dump, "data.pkl"), "wb") as f:
            pickle.dump(data_dump, f)

        print(f"[{get_now_string('%Y-%m-%d %H:%M:%S.%f')}] train set (y_noise) shape: {data_dump['data_train']['y_noise_shape']}")

    def extract_csv(self):
        save_folder = os.path.join(self.args.main_path, self.args.data_dir, self.ode_name, self.args.timestring, "csv")
        dy_path_train_format = os.path.join(save_folder, f"{self.ode_name}_train_{{}}.csv")
        dy_path_test_format = os.path.join(save_folder, f"{self.ode_name}_test_{{}}.csv")

        headers = ["t"] + self.params_config["curve_names"] + ['d'+name for name in self.params_config["curve_names"]]
        assert len(headers) == 2 * self.ode_dim + 1

        print(f"Saving to csv file: {dy_path_train_format} and {dy_path_test_format}")
        for i_env in tqdm(range(self.args.num_env)):
            if self.args.env_id != None and i_env != self.args.env_id:
                continue
            dy_path_train = dy_path_train_format.format(i_env)
            dy_path_test = dy_path_test_format.format(i_env)
            t_col = list(self.t_series_list[i_env][:self.args.n_data_samples])

            y_cols_train, y_cols_test = [], []
            dy_cols_train, dy_cols_test = [], []
            for i_dim in range(self.ode_dim):
                y_col_dim = []
                dy_col_dim = []
                for i_dynamic in self.train_index_list[i_env]:
                    y_col_dim += list(self.y_noise[i_env][i_dynamic][:self.args.n_data_samples, i_dim])
                    dy_col_dim += list(self.dy_noise[i_env][i_dynamic][:self.args.n_data_samples, i_dim])
                y_cols_train.append(y_col_dim)
                dy_cols_train.append(dy_col_dim)

                y_col_dim = []
                dy_col_dim = []
                for i_dynamic in self.test_index_list[i_env]:
                    y_col_dim += list(self.y_noise[i_env][i_dynamic][:self.args.n_data_samples, i_dim])
                    dy_col_dim += list(self.dy_noise[i_env][i_dynamic][:self.args.n_data_samples, i_dim])
                y_cols_test.append(y_col_dim)
                dy_cols_test.append(dy_col_dim)

            t_col_train = t_col * self.num_train_list[i_env]
            t_col_test = t_col * self.num_test_list[i_env]

            train_cols = [t_col_train] + y_cols_train + dy_cols_train
            test_cols = [t_col_test] + y_cols_test + dy_cols_test

            save_to_csv(
                save_path=dy_path_train,
                cols=train_cols,
                headers=headers,
            )
            save_to_csv(
                save_path=dy_path_test,
                cols=test_cols,
                headers=headers,
            )





    # save_to_csv(dy_path_train,
    #                         [t_train] + [y_noise_train[:, j] for j in range(self.ode_dim)] + [dy_noise_train[:, j] for j in range(self.ode_dim_function)],
    #                         ["t"] + self.params_config["curve_names"][:self.ode_dim] + ['d'+name for name in self.params_config["curve_names"][:self.ode_dim_function]])
        # save_to_csv()



    def _plot_dataset(self, save_path, t, t_train, t_test, y, y_train, y_test, y_noise_train, y_noise_test):
        assert len(t_train) == len(y_train) == len(y_noise_train)
        assert len(t_test) == len(y_test) == len(y_noise_test)
        plt.figure(figsize=(16, 9))
        for i in range(self.ode_dim):
            plt.plot(t, y[:, i], label=f"cur-{i + 1}")
        for i in range(self.ode_dim):
            # plt.scatter(t_train, y_train[:, i], label=f"curve-{i + 1} (train)")
            plt.scatter(t_train, y_noise_train[:, i], s=10, label=f"cur-{i + 1} [train noise] [n={len(t_train)}]")
            # plt.scatter(t_test, y_test[:, i], label=f"curve-{i + 1} (test)")
            plt.scatter(t_test, y_noise_test[:, i], s=10, label=f"cur-{i + 1} [test noise] [n={len(t_test)}]")
        plt.xlabel('Time')
        plt.ylabel('Val')
        plt.legend()
        plt.grid()
        plt.savefig(save_path, dpi=300)
        plt.clf()
    
    def _plot_GP(self, save_path, t_train, y_train_gen, y_train_mean, y_train_std):
        assert len(t_train) == len(y_train_mean) == len(y_train_std)
        plt.figure(figsize=(16, 9))
        for i in range(self.ode_dim):
            plt.fill_between(t_train, y_train_mean - y_train_std, y_train_mean + y_train_std, color='gray', alpha=0.5, label='Standard Deviation')
            plt.plot(t_train, y_train_mean, label='Mean', color='blue')
            plt.plot(t_train, y_train_gen, '+g', label='Generated datapoints')
        plt.xlabel('Time')
        plt.ylabel('Val')
        plt.legend()
        plt.grid()
        plt.savefig(save_path, dpi=300)
        plt.clf()

    @staticmethod
    def setup_seed(seed=0):
        np.random.seed(seed)
        random.seed(seed)

    def _save_task_info(self, save_folder):
        # for i_dynamic in range(self.n_dynamic):
        info_path = os.path.join(save_folder, f"{self.ode_name}_info.json")
        # log_truth_ode_list[i_env][task_ode_num - 1]
        task_info = {
            "params": {
                str(env): [x for x in self.params[env]] for env in range(self.args.num_env)
            },
            # "y0": {
            #     str(env): [x for x in self.y0_list[i_dynamic][env]] for env in range(self.args.num_env)
            # },
            "noise_ratio": self.args.noise_ratio,
            "log_truth_ode_list": []
        }

        for i in range(self.args.num_env):
            log_truth_ode = [item.format(*task_info["params"][str(i)]) for item in
                             (self.params_config['truth_ode_format'] if self.params_config.get('truth_ode_format') else [])]
            task_info["log_truth_ode_list"].append(log_truth_ode)
        print(task_info)
        with open(info_path, 'w') as f:
            json.dump(task_info, f, sort_keys=True, indent=4)

if __name__ == "__main__":
    pass
