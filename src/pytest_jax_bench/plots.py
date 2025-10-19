try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError("matplotlib is required for plotting. Please install it via "
                      "'pip install pytest-jax-bench[plot]' or 'pip install matplotlib'.")
import numpy as np
import os
from .data import load_bench_data
import re
from numpy.lib import recfunctions as rfn

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
    if ax is None:
        ax = plt.gca()

    if xaxis == "run":
        commits, first_occ, com_runs_tot, com_of_run = get_commit_info(data)

        x = data["run_id"]

        ax.set_xlabel("Run")

        ax2 = ax.twiny()
        ax2.set_xlim(0, len(data))
        ax2.set_xticks(first_occ)
        ax2.set_xticklabels(commits, fontsize=8, rotation=90)
    elif xaxis == "commit":
        commits, first_occ, com_runs_tot, com_of_run = get_commit_info(data)

        x = com_of_run

        ax.set_xlabel("Commit")
        ax.set_xticks(np.arange(len(commits)), commits, rotation=90 if len(commits) > 10 else 0)
        ax.grid("on")
    elif xaxis == "tag":
        uq_tag = np.unique(data["tag"])
        last_of_tag = np.array([np.max(np.where(data["tag"] == t)[0]) for t in uq_tag])
        data = data[last_of_tag]
        x = np.arange(len(uq_tag))
        # remove [] brackets for parameterized tests_
        lean_tag = [re.sub(r"[\[\]]", "", t) for t in uq_tag]
        tags_are_long = np.any([len(t) > 40/len(lean_tag) for t in lean_tag])
        ax.set_xticks(np.arange(len(lean_tag)), lean_tag, rotation=90 if tags_are_long else 0)
    elif xaxis in data.dtype.names:
        ax.set_xlabel(xaxis)
        x = data[xaxis]
        if (np.max(x) / np.min(x) > 20) and np.all(x > 0):
            ax.set_xscale("log")
    else:
        raise ValueError(f"Unknown xaxis {xaxis}, must be 'commit' or 'run'")
    
    return x, ax, data

def plot_performance(data, title=None, xaxis="commit", ax=None):
    x, ax, data = prepare_xaxis(data, xaxis=xaxis, ax=ax)

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

def plot_performance_tagged(data, title=None, xaxis="commit", ax=None):
    x, ax, data = prepare_xaxis(data, xaxis=xaxis, ax=ax)
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
    x, ax, data = prepare_xaxis(data, xaxis=xaxis, ax=ax)

    ax.set_title(title)
    ax.plot(x, data["jit_peak_bytes"]/1024.**2, label="jit (peak)", marker="o", alpha=0.8)
    if np.any(data["eager_peak_bytes"] >= 0):
        ax.plot(x[data["eager_peak_bytes"]>=0], data["eager_peak_bytes"][data["eager_peak_bytes"]>=0]/1024.**2, label="eager (peak)", marker="o", alpha=0.8)

    ax.plot(x, data["jit_temporary_bytes"]/1024.**2, label="jit (temp)", ls="dashed", marker="o", alpha=0.8)
    if np.any(data["jit_constants_bytes"] > data["jit_peak_bytes"] * 0.01):
        ax.plot(x, data["jit_constants_bytes"]/1024.**2, label="jit (const)", ls="dashed", marker="o", alpha=0.8)

    ax.set_ylabel("Memory (MB)")
    ax.legend()
    if np.any((data["jit_peak_bytes"] > 0) | (data["eager_peak_bytes"] > 0)):
        ax.set_yscale("log")

    return ax

def plot_memory_usage_tagged(data, title=None, xaxis="commit", ax=None):
    x, ax, data = prepare_xaxis(data, xaxis=xaxis, ax=ax)
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

    ax.set_ylabel("Jit-Peak-Memory (MB)")
    ax.legend()
    
    return ax

def find_files(bench_dir="../.benchmarks"):
    files = os.listdir(bench_dir)
    files = [f for f in files if f.endswith(".csv")]
    return [os.path.join(bench_dir, f) for f in files]

def create_and_save(data, path, xaxis="run", tagged=False, save="png", trep=None):
    title = path.split("--")[-1] #.split(":")[-1]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    if tagged:
        plot_performance_tagged(data, title=title, xaxis=xaxis, ax=axs[0])
        plot_memory_usage_tagged(data, title=title, xaxis=xaxis, ax=axs[1])
    else:
        plot_performance(data, title=title, xaxis=xaxis, ax=axs[0])
        plot_memory_usage(data, title=title, xaxis=xaxis, ax=axs[1])
    fig.tight_layout()
    fig.savefig(path + "." + save)
    plt.close(fig)

    if trep == "print":
        print(f"Generated plot {path}.{save}")
    elif trep is not None:
        trep.write_line(f"Generated plot {path}.{save}")

def plot_normal_benchmark(path, xaxis="run", save="png", trep=None):
    data = load_bench_data(path + ".csv")
    if data is None: return

    if len(np.unique(data["tag"])) > 1:
        create_and_save(data, path=path, xaxis=xaxis, tagged=True, save=save, trep=trep)
        create_and_save(data, path=path + "_xtag", xaxis="tag", tagged=False, save=save, trep=trep)
    else:
        create_and_save(data, path=path, xaxis=xaxis, tagged=False, save=save, trep=trep)

def plot_parametrized_benchmark(path, xaxis="commit", save="png", trep=None):
    data, pars = load_bench_data(path, interprete_parameters=True)
    if data is None: return

    data_with_par =  rfn.merge_arrays((data, pars), flatten=True, usemask=False)
    
    for k in pars.dtype.names:
        # Find the last locations of unique pairs of tag and k
        pairs = data_with_par[["tag", k]]
        last_idx = len(data_with_par) - 1 - np.unique(pairs[::-1], return_index=True)[1]

        if len(np.unique(data["tag"])) > 1:
            create_and_save(data_with_par[last_idx], path=path + f"/{k}", xaxis=k, tagged=True, save=save, trep=trep)
        else:
            create_and_save(data_with_par[last_idx], path=path + f"/{k}", xaxis=k, tagged=False, save=save, trep=trep)

def plot_all_benchmarks(bench_dir="../.benchmarks", xaxis="commit", save="png", trep=None):
    files = os.listdir(bench_dir)
    files_csv = [f for f in files if f.endswith(".csv")]

    dirs = [f for f in files if os.path.isdir(os.path.join(bench_dir, f))]
    dirs_csv = []

    for d in dirs:
        sub_files = os.listdir(os.path.join(bench_dir, d))
        sub_csv = [f for f in sub_files if f.endswith(".csv")]
        files_csv.extend( [os.path.join(d, f) for f in sub_csv] )
        if len(sub_csv) > 0:
            dirs_csv.append(d)

    for f in files_csv:
        path = os.path.join(bench_dir, f.strip(".csv"))
        plot_normal_benchmark(path, xaxis=xaxis, save=save, trep=trep)
    for d in dirs_csv:
        path = os.path.join(bench_dir, d)
        plot_parametrized_benchmark(path, xaxis=xaxis, save=save, trep=trep)