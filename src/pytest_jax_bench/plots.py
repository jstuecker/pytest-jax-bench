try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError("matplotlib is required for plotting. Please install it via "
                      "'pip install pytest-jax-bench[plot]' or 'pip install matplotlib'.")
import numpy as np
import os
from .data import load_bench_data
import re

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
        ax2.set_xticklabels(commits, fontsize=8, rotation=90)
    elif xaxis == "commit":
        x = com_of_run

        ax.set_xlabel("Commit")
        ax.set_xticks(np.arange(len(commits)), commits, rotation=90 if len(commits) > 10 else 0)
        ax.grid("on")
    else:
        raise ValueError(f"Unknown xaxis {xaxis}, must be 'commit' or 'run'")
    
    return x, ax

def plot_run_performance(data, title=None, xaxis="commit", ax=None):
    if len(np.unique(data["tag"])) > 1:
        return plot_run_performance_tagged(data, title=title, xaxis=xaxis, ax=ax)

    x, ax = prepare_xaxis(data, xaxis=xaxis, ax=ax)

    ax.set_title(title)
    if np.any(data["jit_mean_ms"] > 0):
        ax.plot(x, data["jit_mean_ms"], marker="o", label="jitted", alpha=0.8)
        ax.fill_between(x, data["jit_mean_ms"]-data["jit_std_ms"], data["jit_mean_ms"]+data["jit_std_ms"], alpha=0.3)

    if np.any(data["eager_mean_ms"] > 0):
        ax.plot(x, data["eager_mean_ms"], marker="o", label="eager", alpha=0.8)
        ax.fill_between(x, data["eager_mean_ms"]-data["eager_std_ms"], data["eager_mean_ms"]+data["eager_std_ms"], alpha=0.3)

    ax.plot(x, data["compile_ms"], marker="o", label="compile", alpha=0.8)
    
    ax.set_ylabel("Time (ms)")
    ax.legend()
    if np.any((data["jit_mean_ms"] > 0) | (data["eager_mean_ms"] > 0)):
        ax.set_yscale("log")

    return ax

def plot_run_performance_tagged(data, title=None, xaxis="commit", ax=None):
    x, ax = prepare_xaxis(data, xaxis=xaxis, ax=ax)
    ax.set_title(title)

    tags = np.unique(data["tag"])
    for i,tag in enumerate(tags):
        data_t = data[data["tag"] == tag]
        xt = x[data["tag"] == tag]

        color = f"C{i}"
        ax.plot(xt, data_t["jit_mean_ms"], marker="o", label=tag, alpha=0.8, color=color)
        ax.fill_between(xt, data_t["jit_mean_ms"]-data_t["jit_std_ms"], data_t["jit_mean_ms"]+data_t["jit_std_ms"], alpha=0.6, color=color)

        ax.plot(xt, data_t["eager_mean_ms"], marker="o", alpha=0.2, color=color, ls="dashed")
        ax.fill_between(xt, data_t["eager_mean_ms"]-data_t["eager_std_ms"], data_t["eager_mean_ms"]+data_t["eager_std_ms"], alpha=0.2, color=color)
        
        if np.any((data_t["jit_mean_ms"] > 0) | (data_t["eager_mean_ms"] > 0)):
            ax.set_yscale("log")
    
    if np.any(data["jit_mean_ms"] > 0):
        ax.plot([], [], label="jitted", color="k", marker="o", alpha=0.8)
    if np.any(data["eager_mean_ms"] > 0):
        ax.plot([], [], label="eager", color="k", marker="o", alpha=0.2, ls="dashed")

    ax.set_ylabel("Time (ms)")
    ax.legend()

    return ax

def plot_memory_usage(data, title=None, xaxis="commit", ax=None):
    if len(np.unique(data["tag"])) > 1:
        return plot_memory_usage_tagged(data, title=title, xaxis=xaxis, ax=ax)

    x, ax = prepare_xaxis(data, xaxis=xaxis, ax=ax)

    ax.set_title(title)
    ax.plot(x, data["jit_peak_bytes"]/1024.**2, label="jit (peak)", marker="o", alpha=0.8)
    ax.plot(x[data["eager_peak_bytes"]>=0], data["eager_peak_bytes"][data["eager_peak_bytes"]>=0]/1024.**2, label="eager (peak)", marker="o", alpha=0.8)

    ax.plot(x, data["jit_temporary_bytes"]/1024.**2, label="jit (temp)", ls="dashed", marker="o", alpha=0.8)
    if np.any(data["jit_constants_bytes"] > 1e3):
        ax.plot(x, data["jit_constants_bytes"]/1024.**2, label="jit (const)", ls="dashed", marker="o", alpha=0.8)

    ax.set_ylabel("Memory (MB)")
    ax.legend()
    if np.any((data["jit_peak_bytes"] > 0) | (data["eager_peak_bytes"] > 0)):
        ax.set_yscale("log")

    return ax

def plot_memory_usage_tagged(data, title=None, xaxis="commit", ax=None):
    x, ax = prepare_xaxis(data, xaxis=xaxis, ax=ax)
    ax.set_title(title)

    tags = np.unique(data["tag"])
    for i,tag in enumerate(tags):
        data_t = data[data["tag"] == tag]
        xt = x[data["tag"] == tag]
        label = tag if tag != "base" else ""

        color = f"C{i}"

        ax.plot(xt, data_t["jit_peak_bytes"]/1024.**2, label=f"{label}", marker="o", alpha=0.8, color=color)
        # So far, eager memory doesn't work well with tags. Disable for now.
        # ax.plot(xt, data_t["eager_peak_bytes"]/1024.**2, marker="x", alpha=0.8, color=color, ls="dashed")
        if np.any((data["jit_peak_bytes"] > 0) | (data["eager_peak_bytes"] > 0)):
            ax.set_yscale("log")

    # ax.plot([], [], label="jit (peak)", color="k", marker="o")
    # ax.plot([], [], label="eager (peak)", ls="dashed", color="k", marker="x")

    ax.set_ylabel("Jit-Peak-Memory (MB)")
    ax.legend()
    

    return ax


def find_files(bench_dir="../.benchmarks"):
    files = os.listdir(bench_dir)
    files = [f for f in files if f.endswith(".csv")]
    return [os.path.join(bench_dir, f) for f in files]

def get_data_of_parameterized_group(files, path_group_base):
    files_group = tuple(filter(lambda f: path_group_base in f, files))
    data = []
    for file in files_group:
        pars = re.search(r"\[(.*)\]", file.replace(".csv","")).group(0)
        d = load_bench_data(file)
        if np.all(d["tag"] == "base"):
            d["tag"] = pars
        else:
            d["tag"] += pars
        data.append(d)
    data = np.concatenate(data)
    return data

def iterate_data(paths=None, bench_dir=".benchmarks", group_par=False):
    files = find_files(bench_dir)

    if paths is None:
        paths = [file.replace(".csv", "") for file in files]
    
    groups_done = []
    for path in paths:
        if not os.path.isfile(path + ".csv"):
            continue

        if group_par and re.search(r"\[.*\]$", path) is not None:
            # for parameterized groups we may want to summarize several files
            path = re.sub(r"\[.*\]$", "", path)
            if path in groups_done:
                continue

            data = get_data_of_parameterized_group(files, path)
            groups_done.append(path)
        else:
            data = load_bench_data(path + ".csv")
        title = path.split("/")[-1].split(":")[-1]

        yield data, title, path

def plot_all_benchmarks_together(paths=None, bench_dir=".benchmarks", xaxis="commit", save="png", group_par=False):
    n = len(paths)
    fig, axs = plt.subplots(n, 2, figsize=(10, 4*n))

    for i, (data, title, path) in enumerate(iterate_data(paths, bench_dir=bench_dir, group_par=group_par)):
        plot_run_performance(data, title=title, xaxis=xaxis, ax=axs[i,0])
        plot_memory_usage(data, title=title, xaxis=xaxis, ax=axs[i,1])

    # delete unused axes if any
    if i+1 < len(axs):
        for j in range(i+1, len(axs)):
            fig.delaxes(axs[j,0])
            fig.delaxes(axs[j,1])

    fig.tight_layout()

    if save:
        assert save in {"png", "pdf"}
        fig.savefig(os.path.join(bench_dir, "all_benchmarks.%s" % save), bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return fig, axs


def plot_all_benchmarks_individually(paths=None, bench_dir=".benchmarks", xaxis="commit", save="png", group_par=False):
    figs = []
    for data, title, path in iterate_data(paths, bench_dir=bench_dir, group_par=group_par):
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        plot_run_performance(data, title=title, xaxis=xaxis, ax=axs[0])
        plot_memory_usage(data, title=title, xaxis=xaxis, ax=axs[1])
        fig.tight_layout()
        figs.append((fig, axs))
        
        if save:
            assert save in {"png", "pdf"}
            fig.savefig(path + "." + save)
            plt.close(fig)
        else:
            plt.show()