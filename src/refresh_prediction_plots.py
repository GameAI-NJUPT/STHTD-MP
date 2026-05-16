import importlib.util
import os

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
EXPERIMENT_SCRIPT = os.path.join(SCRIPT_DIR, "paper_prediction_experiments.py")

spec = importlib.util.spec_from_file_location("paper_prediction_experiments", EXPERIMENT_SCRIPT)
experiments = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiments)


def main():
    base_dir = os.path.join(ROOT_DIR, "paper_results", "prediction")
    main_algorithms = [
        experiments.TD,
        experiments.GTD2,
        experiments.TDC,
        experiments.TDRC,
        experiments.GTD2_MP,
        experiments.HTD,
        experiments.ETD0,
        experiments.STHTD,
        experiments.STHTD_MP,
    ]
    damped_algorithms = [
        experiments.STHTD,
        experiments.STHTD_MP,
        experiments.STHTD_DAMPED,
        experiments.STHTD_MP_DAMPED,
    ]
    for env_name in ["two_state", "baird", "random_walk", "boyan_chain"]:
        out_dir = os.path.join(base_dir, env_name)
        curves_path = os.path.join(out_dir, "curves.npz")
        curves = np.load(curves_path)
        results = {name: curves[name] for name in main_algorithms if name in curves.files}
        experiments.plot_results(out_dir, env_name, results, experiments.STHTD_MP)
        damped_results = {name: curves[name] for name in damped_algorithms if name in curves.files}
        if len(damped_results) > 2:
            experiments.plot_results(out_dir, env_name, damped_results, experiments.STHTD_MP, suffix="damped_appendix")
        print(f"refreshed {env_name}")


if __name__ == "__main__":
    main()
