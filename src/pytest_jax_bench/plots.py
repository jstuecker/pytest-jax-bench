import matplotlib.pyplot as plt
import numpy as np
import os
from .data import load_bench_data

def get_commit_info(arr):
    # First get each commit once. Note this will screw up the order, we'll fix it later
    commits = np.sort(np.unique(arr["commit"]))

    # per commit info
    first_occ = np.array([np.min(np.where(arr["commit"] == c)[0]) for c in commits])
    com_runs_tot = np.array([np.max(arr["commit_run"][arr["commit"] == c]) for c in commits])

    # per run info
    com_of_run = np.searchsorted(commits, arr["commit"])

    # restore order
    isort = np.argsort(first_occ)
    inv = np.zeros(len(commits), dtype=int)
    inv[isort] = np.arange(len(isort))

    commits = commits[isort]
    first_occ = first_occ[isort]
    com_runs_tot = com_runs_tot[isort]
    com_of_run = inv[com_of_run]

    # add intra-commmit offset
    com_of_run = com_of_run + arr["commit_run"] / (1 + com_runs_tot[com_of_run])

    return commits, first_occ, com_runs_tot, com_of_run

def prepare_xaxis(data, xaxis="commit", ax=None):
    commits, first_occ, com_runs_tot, com_of_run = get_commit_info(data)

    if ax is None:
        ax = plt.gca()

    if xaxis == "run":
        x = data["run_id"]

        ax.set_xlabel("Run")

        ax2 = ax.twiny()
        ax2.set_xlim(0, len(data))
        ax2.set_xticks(first_occ)
        ax2.set_xticklabels(commits, fontsize=8, rotation=90 if len(commits) > 10 else 0)
    elif xaxis == "commit":
        x = com_of_run

        ax.set_xlabel("Commit")
        ax.set_xticks(np.arange(len(commits)), commits, rotation=90 if len(commits) > 10 else 0)
        ax.grid("on")
    else:
        raise ValueError(f"Unknown xaxis {xaxis}, must be 'commit' or 'run'")
    
    return x, ax

def plot_run_performance(data, title=None, xaxis="commit", ax=None):
    x, ax = prepare_xaxis(data, xaxis=xaxis, ax=ax)

    ax.set_title(title)
    ax.plot(x, data["jit_mean_ms"], marker="o", label="jitted", alpha=0.8)
    ax.fill_between(x, data["jit_mean_ms"]-data["jit_std_ms"], data["jit_mean_ms"]+data["jit_std_ms"], alpha=0.3)

    ax.plot(x, data["eager_mean_ms"], marker="o", label="eager", alpha=0.8)
    ax.fill_between(x, data["eager_mean_ms"]-data["eager_std_ms"], data["eager_mean_ms"]+data["eager_std_ms"], alpha=0.3)
    
    ax.set_ylabel("Time (ms)")
    ax.legend()
    if np.any((data["jit_mean_ms"] > 0) | (data["eager_mean_ms"] > 0)):
        ax.set_yscale("log")

    return ax

def plot_memory_usage(data, title=None, xaxis="commit", ax=None):
    x, ax = prepare_xaxis(data, xaxis=xaxis, ax=ax)

    ax.set_title(title)
    ax.plot(x, data["jit_peak_bytes"], label="jit (peak)", marker="o", alpha=0.8)
    ax.plot(x, data["eager_peak_memory"], label="eager (peak)", marker="o", alpha=0.8)

    ax.plot(x, data["jit_temporary_bytes"], label="jit (temp)", ls="dashed", marker="o", alpha=0.8)
    ax.plot(x, data["jit_constants_bytes"], label="jit (const)", ls="dashed", marker="o", alpha=0.8)

    ax.set_ylabel("Memory (MB)")
    ax.legend()
    if np.any((data["jit_peak_bytes"] > 0) | (data["eager_peak_memory"] > 0)):
        ax.set_yscale("log")

    return ax

def find_files(bench_dir="../.benchmarks"):
    files = os.listdir(bench_dir)
    files = [f for f in files if f.endswith(".csv")]
    return [os.path.join(bench_dir, f) for f in files]

def plot_all_benchmarks_together(bench_dir="../.benchmarks", xaxis="commit", save="png"):
    files = find_files(bench_dir)

    n = len(files)
    fig, axs = plt.subplots(n, 2, figsize=(10, 4*n))

    for i, file in enumerate(files):
        data = load_bench_data(file)
        title = os.path.basename(file).replace(".csv", "")
        plot_run_performance(data, title=title, xaxis=xaxis, ax=axs[i,0])
        plot_memory_usage(data, title=title, xaxis=xaxis, ax=axs[i,1])

    fig.tight_layout()

    if save:
        assert save in {"png", "pdf"}
        fig.savefig(os.path.join(bench_dir, "all_benchmarks.%s" % save))
        plt.close(fig)

    return fig, axs

def plot_all_benchmarks_individually(bench_dir="../.benchmarks", xaxis="commit", save="png"):
    assert save

    files = find_files(bench_dir)
    
    figs = []
    for file in files:
        data = load_bench_data(file)
        title = os.path.basename(file).replace(".csv", "")
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        plot_run_performance(data, title=title, xaxis=xaxis, ax=axs[0])
        plot_memory_usage(data, title=title, xaxis=xaxis, ax=axs[1])
        fig.tight_layout()
        figs.append((fig, axs))
        
        if save:
            assert save in {"png", "pdf"}
            fig.savefig(os.path.join(bench_dir, title + "." + save))
            plt.close(fig)