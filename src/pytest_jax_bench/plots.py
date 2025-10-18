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
    elif xaxis == "tag":
        uq_tag = np.unique(data["tag"])
        last_of_tag = np.array([np.max(np.where(data["tag"] == t)[0]) for t in uq_tag])
        data = data[last_of_tag]
        x = np.arange(len(uq_tag))
        # remove [] brackets for parameterized tests_
        lean_tag = [re.sub(r"[\[\]]", "", t) for t in uq_tag]
        tags_are_long = np.any([len(t) > 100/len(lean_tag) for t in lean_tag])
        ax.set_xticks(np.arange(len(lean_tag)), lean_tag, rotation=90 if tags_are_long else 0)
    else:
        raise ValueError(f"Unknown xaxis {xaxis}, must be 'commit' or 'run'")
    
    return x, ax, data

def plot_run_performance(data, title=None, xaxis="commit", ax=None):
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

def plot_run_performance_tagged(data, title=None, xaxis="commit", ax=None):
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

def iterate_data(paths=None, bench_dir=".benchmarks"):
    files = find_files(bench_dir)

    if paths is None:
        paths = [file.replace(".csv", "") for file in files]
    
    groups_done = []
    for path in paths:
        if not os.path.isfile(path + ".csv"):
            continue

        data = load_bench_data(path + ".csv")
        yield data, path

        if re.search(r"\[.*\]$", path) is not None:
            # for parameterized groups we return a second time together
            path_gr = re.sub(r"\[.*\]$", "", path)
            if path_gr in groups_done:
                continue

            data_gr = get_data_of_parameterized_group(files, path_gr)
            groups_done.append(path_gr)

            yield data_gr, path_gr

def plot_all_benchmarks(paths=None, bench_dir=".benchmarks", xaxis="commit", save="png", trep=None):
    assert save in {"png", "pdf"}

    def save_plot(data, path, xaxis=xaxis, tagged=False):
        title = path.split("/")[-1].split(":")[-1]
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        if tagged:
            plot_run_performance_tagged(data, title=title, xaxis=xaxis, ax=axs[0])
            plot_memory_usage_tagged(data, title=title, xaxis=xaxis, ax=axs[1])
        else:
            plot_run_performance(data, title=title, xaxis=xaxis, ax=axs[0])
            plot_memory_usage(data, title=title, xaxis=xaxis, ax=axs[1])
        fig.tight_layout()
        fig.savefig(path + "." + save)
        plt.close(fig)

        if trep == "print":
            print(f"Generated plot {path}.{save}")
        elif trep is not None:
            trep.write_line(f"Generated plot {path}.{save}")

    for data, path in iterate_data(paths, bench_dir=bench_dir):
        if len(np.unique(data["tag"])) > 1:
            save_plot(data, path=path, xaxis=xaxis, tagged=True)
            save_plot(data, path=path + "_tag", xaxis="tag", tagged=False)
        else:
            save_plot(data, path=path, xaxis=xaxis, tagged=False)