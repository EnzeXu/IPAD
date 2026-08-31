# IPAD

IPAD discovers closed-form governing equations that stay **invariant across multiple environments** while letting per-environment coefficients vary. Given noisy trajectories from several environments of the same physical system, IPAD searches for a single equation skeleton (shared structure) whose coefficients best explain every environment simultaneously.

This repository provides a runnable implementation of IPAD's core discovery pipeline (dataset generation + multi-environment symbolic regression search) for two dynamical systems:

- Lotka-Volterra
- SIR

---

## Table of Contents

1. [Authors](#authors)
2. [Repository Structure](#repository-structure)
3. [Getting Started](#getting-started)
4. [Usage](#usage)
5. [Citing This Work](#citing-this-work)
6. [License](#license)
7. [Questions](#questions)

---

## Authors

- **Enze Xu**: [exu03@wm.edu](mailto:exu03@wm.edu)
- **Toon Tran**
- **Hongjue Zhao**
- **Yuchen Wang**
- **Dr. Mengdi Huai**
- **Dr. Huajie Shao**: [hshao@wm.edu](mailto:hshao@wm.edu)

---

## Repository Structure

- **`invariant_physics/`**: the core library.
  - **`dataset/`**: dataset generation for the 2 supported systems (`Lotka_Volterra`, `SIR`), argument parsing, and symbolic-expression utilities.
  - **`loss/`**: the trajectory-fitting loss used to score candidate equations (`VF_Loss`).
  - **`spl/`**: the Symbolic Physics Learner (MCTS-style grammar search) that discovers equations, and the multi-environment reward functions.
- **`run.py`**: main entry point — generates a fresh dataset and runs the multi-environment equation search.
- **`make_datasets.py`**: optional helper to pre-generate a dataset once and reuse it across multiple `run.py` invocations (see [Usage](#usage)).
- **`requirements.txt`**: pinned Python dependencies.
- **`LICENSE`**: the license governing the use of this code.

Two folders are kept empty (via `.gitignore`) but must exist for `run.py` to write into: `outputs/` and `outputs_reward_his/`. Generated datasets (`data/`) and run logs (`logs/`) are created automatically and are not committed.

---

## Getting Started

This project requires Python 3.12+ (pinned dependency versions in `requirements.txt` need it) and is compatible with macOS, Linux, and Windows.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/EnzeXu/IPAD.git
   cd IPAD
   ```

2. **Create and activate a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Quick start

`run.py` builds its own dataset on the fly, so a single command is enough to see the whole pipeline run end to end. The default arguments are already set to a configuration that reliably recovers the correct equation skeleton for every equation of both supported systems, at both noise-free and noisy (5%) settings:

```bash
python run.py --task Lotka_Volterra
python run.py --task SIR
```

This is equivalent to running with the arguments spelled out explicitly:

```bash
python run.py --task Lotka_Volterra --num_env 5 --task_ode_num 1 --loss_func VF \
    --num_run 2 --transplant_step 500 --n_dynamic 40/40/40/40/40 --eta 0.9999 \
    --combine_operator average
```

- `--task` selects the system: `Lotka_Volterra` or `SIR`.
- `--task_ode_num` selects which output equation to discover (1-indexed). `Lotka_Volterra` has 2 equations (`1` = `dx/dt`, `2` = `dy/dt`); `SIR` has 3 (`1`, `2`, `3`).
- `--num_env` is the number of environments used simultaneously to enforce invariance.
- `--noise_ratio` (default `0.0`) adds observation noise as a fraction of each trajectory's std-dev; `--noise_ratio 0.05` was also verified to still recover the correct skeleton for every equation of both systems.

Each run takes roughly 1–3 minutes on a modern multi-core machine. The discovered equation, the ground-truth equation, and match statistics are printed at the end of the run (see the summary block), and a machine-readable summary row is appended to `logs/summary/logs_{task}_end.csv`.

Run `python run.py --help` for the full list of arguments (reward variants, integration method, train/val/test split, etc.).

### Reusing a fixed dataset across runs

By default, every `run.py` call generates its own fresh dataset. To evaluate multiple configurations on the *same* dataset instead, generate it once with `make_datasets.py` and a fixed `--timestring`, then reuse it:

```bash
python make_datasets.py --task Lotka_Volterra --num_env 5 --n_dynamic 40/40/40/40/40 --timestring 20260101_000000_000000
python run.py --task Lotka_Volterra --num_env 5 --task_ode_num 1 --n_dynamic 40/40/40/40/40 \
    --timestring 20260101_000000_000000 --load_data_from_existing 1
```

---

## Citing This Work

If you use this code for academic research, please cite:

```bibtex
@article{xu2026identifying,
  title     = {Identifying Invariant Physical Dynamics Across Multiple Environments},
  author    = {Xu, Enze and Tran, Toon and Zhao, Hongjue and Wang, Yuchen and Huai, Mengdi and Shao, Huajie},
  journal   = {Transactions on Machine Learning Research},
  year      = {2026},
  month     = {September}
}
```

---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

---

## Questions

For any questions or contributions, please feel free to open an issue or submit a pull request. Alternatively, you can contact us directly at [exu03@wm.edu](mailto:exu03@wm.edu).
