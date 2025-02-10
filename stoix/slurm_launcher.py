import itertools
import os
import subprocess

import hydra
import submitit
from omegaconf import DictConfig


def run_experiment(algorithm_exec_file: str, environment: str, seed: int) -> None:
    """
    Runs a single Stoix experiment via a subprocess run.

    Args:
        algorithm_exec_file: Algorithm/system (e.g. 'dqn', 'ppo') exec file.
            e.g. 'stoix/systems/ppo/anakin/ff_ppo.py'
        environment: Environment config (e.g. 'gymnax/cartpole or brax/ant')
        seed: Random seed for reproducibility
    """

    cmd = f"python {algorithm_exec_file} env={environment} arch.seed={seed}"

    print(f"Running command: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def filter_none_values(d: dict) -> dict:
    """
    Returns a new dictionary containing only the items from the input dictionary
    where the value is not None.

    Args:
        d: The input dictionary.
    Returns:
        A dictionary with keys whose values are not None.
    """
    return {key: value for key, value in d.items() if value is not None}


@hydra.main(version_base="1.2", config_path="./configs/launcher", config_name="slurm")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for launching multiple Stoix experiments on SLURM-based cluster.

    Args:
        cfg: The Hydra-populated configuration object.
    """
    # Create the submitit executor for SLURM.
    executor = submitit.AutoExecutor(folder=cfg.slurm.folder)

    # Build SLURM parameter dictionary. Only pass parameters that are non-None.
    # If you pass None to some fields, submitit may ignore them or raise an error,
    # so we filter them out where appropriate.
    slurm_params = {
        "nodes": cfg.slurm.nodes,
        "gpus_per_node": cfg.slurm.gpus_per_node,
        "cpus_per_task": cfg.slurm.cpus_per_task,
        "time": cfg.slurm.time,
        "chdir": os.getcwd(),
        "slurm_account": cfg.slurm.account,
        "slurm_qos": cfg.slurm.qos,
        "slurm_partition": cfg.slurm.partition,
    }
    slurm_params = filter_none_values(slurm_params)

    # Update the executor with SLURM parameters
    executor.update_parameters(
        slurm_job_name=cfg.experiment_group, slurm_additional_parameters=slurm_params
    )

    # Prepare the Cartesian product of algorithm_execs, environments, seeds.
    jobs = []
    with executor.batch():
        for algorithm_exec, env, seed in itertools.product(
            cfg.experiment.algorithm_exec_files, cfg.experiment.environments, cfg.experiment.seeds
        ):
            print(f"Submitting job for {algorithm_exec.split('/')[-1]} on {env} with seed {seed}.")
            job = executor.submit(run_experiment, algorithm_exec, env, seed)
            jobs.append(job)


if __name__ == "__main__":
    main()
