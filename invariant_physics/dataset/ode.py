import os
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.integrate import odeint, solve_ivp
from tqdm import tqdm

from ._utils import sample_lhs, save_to_csv, params_random, generate_ordered_indices, get_now_string


class ODEDataset:
    def __init__(self, args, params_config, non_ode_function=False):
        self.args = args
        self.setup_seed(self.args.seed)
        self.params_config = params_config
        self.n_dynamic_list = self.args.n_dynamic_list
        self.ode_dim = self.params_config["ode_dim"]
        self.ode_dim_function = self.params_config["ode_dim_function"]
        self.ode_name = self.params_config["task"]

        self.params, self.y0_list = self._get_ode_params_and_y0(self.args.num_env, self.args.params_strategy)
        print("self.params.shape:", [item.shape for item in self.params])
        print("self.y0_list.shape:", [item.shape for item in self.y0_list])
        self.params = self._set_partial_param(self.params, self.args.partial_mask_list)  # For partial masked odes
        self.t_series_list = []
        self.num_train_list, self.num_val_list, self.num_test_list = [], [], []
        self.train_index_list, self.val_index_list, self.test_index_list = [], [], []

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

        self.non_ode_function = non_ode_function  # It's a direct function, not an ODE

    def _set_partial_param(self, params, mask):
        assert len(params) == len(mask) == self.args.num_env, f"len(params)={len(params)}, len(mask)={len(mask)}, self.args.num_env={self.args.num_env} should be the same!"
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
                self.t_series_list.append(np.asarray([self.params_config["t_min"] + self.dt * j for j in range(self.N)]))
            else:  # lhs
                self.t_series_list.append(sample_lhs(self.params_config["t_min"], self.params_config["t_max"], self.N))

    def build(self):
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

        save_folder = os.path.join(self.args.main_path, self.args.data_dir, self.ode_name, self.args.timestring, "csv")
        save_folder_dump = os.path.join(self.args.main_path, self.args.data_dir, self.ode_name, self.args.timestring, "dump")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if not os.path.exists(save_folder_dump):
            os.makedirs(save_folder_dump)

        print(f"ode_name: {self.ode_name}")
        print(f"num_env: {self.args.num_env}")
        print(f"t_min: {self.params_config['t_min']}, t_max: {self.params_config['t_max']}, dt: {self.params_config['dt']}, N: {self.N}")
        print(f"train set: {self.num_train_list} ({self.args.train_ratio * 100:.1f} %), val set: {self.num_val_list} ({self.args.val_ratio * 100:.1f} %), test set: {self.num_test_list} ({self.args.test_ratio * 100:.1f} %), total: {self.args.n_dynamic}, N: {self.N}")
        print(f"noise ratio: {self.args.noise_ratio}")
        print(f"save_folder: {save_folder}")

        self._save_task_info(save_folder)

        for i in range(self.args.num_env):
            print(f"Environment {i:02d}: params={[f'{item:.8f}' for item in self.params[i]]}, truth = {[item.format(*self.params[i]) for item in (self.params_config['truth_ode_format'] if self.params_config.get('truth_ode_format') else [])]}")

        for i in tqdm(range(self.args.num_env)):
            if self.args.env_id != None and i != self.args.env_id:
                continue
            assert self.args.integrate_method in ["ode_int", "solve_ivp"]
            if not self.non_ode_function:
                for i_dynamic in range(self.n_dynamic_list[i]):
                    if self.args.integrate_method == "ode_int":
                        self.y[i][i_dynamic] = odeint(self._func, self.y0_list[i][i_dynamic], self.t_series_list[i], (i,))
                    else:
                        sol = solve_ivp(fun=self._func_solve_ivp, t_span=(self.t_series_list[i][0], self.t_series_list[i][-1]), y0=np.asarray(self.y0_list[i][i_dynamic]), args=(i,), t_eval=self.t_series_list[i], method="RK45")
                        self.y[i][i_dynamic] = np.transpose(sol.y, (1, 0))
            else:
                assert NotImplementedError

            for i_dynamic in range(self.n_dynamic_list[i]):
                std_base = np.std(self.y[i][i_dynamic], axis=0)
                noise_sigma = std_base * self.args.noise_ratio
                self.y_noise[i][i_dynamic] = self.y[i][i_dynamic] + noise_sigma * np.random.randn(*self.y[i][i_dynamic].shape)

                y_noise_train = self.y_noise[i][i_dynamic]
                y_noise_test = self.y_noise[i][i_dynamic]

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
            index_list = getattr(self, f"{data_type}_index_list")

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
            data_dump[f"data_{data_type}"] = one_type_data_dump
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

    def _plot_dataset(self, save_path, t, t_train, t_test, y, y_train, y_test, y_noise_train, y_noise_test):
        assert len(t_train) == len(y_train) == len(y_noise_train)
        assert len(t_test) == len(y_test) == len(y_noise_test)
        plt.figure(figsize=(16, 9))
        for i in range(self.ode_dim):
            plt.plot(t, y[:, i], label=f"cur-{i + 1}")
        for i in range(self.ode_dim):
            plt.scatter(t_train, y_noise_train[:, i], s=10, label=f"cur-{i + 1} [train noise] [n={len(t_train)}]")
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
        info_path = os.path.join(save_folder, f"{self.ode_name}_info.json")
        task_info = {
            "params": {
                str(env): [x for x in self.params[env]] for env in range(self.args.num_env)
            },
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
