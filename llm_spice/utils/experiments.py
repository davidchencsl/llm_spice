import itertools
import multiprocess as mp
import traceback
import dill as pickle
import os
import inspect

from llm_spice.utils.common import PROJECT_ROOT

from tqdm import tqdm


EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")


def gen_args(configs):
    finalized_experiments = []
    for combo in itertools.product(*configs.values()):
        exp_config = {}
        for key, value in zip(configs.keys(), combo):
            exp_config[key] = value
        finalized_experiments.append(exp_config)
    return finalized_experiments


def parallel_experiment_runner(function=None, configs=None, overwrite=False, name=None):
    if name is None:
        name = os.path.splitext(os.path.basename(inspect.stack()[1].filename))[0]
    pickle_path = os.path.join(EXPERIMENTS_DIR, f"data/pkl/{name}.pkl")
    if os.path.exists(pickle_path) and not overwrite:
        with open(pickle_path, "rb") as f:
            return pickle.load(f)

    assert function and configs, "function and configs must be provided if cache miss"

    def run_experiment_wrapper(kwargs):
        try:
            return function(**kwargs)
        except Exception as e:
            print(f"Error running {kwargs}: {e}")
            traceback.print_exc()
            return None

    args = gen_args(configs)

    with mp.Pool(initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as executor:  # type: ignore
        stats = list(
            tqdm(
                executor.imap_unordered(run_experiment_wrapper, args),
                total=len(args),
                position=0,
                desc=f"Running {name}",
            )
        )

    os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
    with open(pickle_path, "wb") as f:
        pickle.dump(stats, f)
    return stats


def mp_tqdm(*args, **kwargs):
    idx = mp.current_process()._identity[0] - 1 if mp.current_process()._identity else 0  # type: ignore
    return tqdm(*args, **kwargs, position=idx + 1, desc=f"Process {idx}", leave=False)
