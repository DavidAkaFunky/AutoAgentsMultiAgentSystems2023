import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

def z_table(confidence):
    """Hand-coded Z-Table

    Parameters
    ----------
    confidence: float
        The confidence level for the z-value.

    Returns
    -------
        The z-value for the confidence level given.
    """
    return {
        0.99: 2.576,
        0.95: 1.96,
        0.90: 1.645
    }[confidence]


def confidence_interval(mean, n, confidence):
    """Computes the confidence interval of a sample.

    Parameters
    ----------
    mean: float
        The mean of the sample
    n: int
        The size of the sample
    confidence: float
        The confidence level for the z-value.

    Returns
    -------
        The confidence interval.
    """
    return z_table(confidence) * (mean / math.sqrt(n))


def standard_error(std_dev, n, confidence):
    """Computes the standard error of a sample.

    Parameters
    ----------
    std_dev: float
        The standard deviation of the sample
    n: int
        The size of the sample
    confidence: float
        The confidence level for the z-value.

    Returns
    -------
        The standard error.
    """
    return z_table(confidence) * (std_dev / math.sqrt(n))

def compare_results_pop(results, filename, confidence=0.95, title="Agents Comparison", colors=None, plot=False, metric="Population"):

    """Displays a bar plot comparing the performance of different agents/teams.

        Parameters
        ----------

        results: dict
            A dictionary where keys are the names and the values sequences of trials
        confidence: float
            The confidence level for the confidence interval
        title: str
            The title of the plot
        metric: str
            The name of the metric for comparison
        colors: Sequence[str]
            A sequence of colors (one for each agent/team)

        """
    situations = list(results.keys())
    results = list(results.values())
    names = range(0, len(results[0]), len(results[0]) // 20)
    plt.figure().set_figwidth(10)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}')) # 1 decimal places
    for i in range(len(results)):
        results_i = results[i]
        means = [result.mean() for result in results_i]
        std_devs = [result.std() for result in results_i]
        N = [result.size for result in results_i]
        errors = [standard_error(std_devs[j], N[j], confidence) for j in range(len(means))]
        if plot:
            # Uncomment to get plot points
            # plt.plot(x_pos, means, "bo", color=colors[i] if colors is not None else "gray")
            plt.errorbar(range(len(results_i)), means, errors, capsize=3, linewidth=1, color=colors[i] if colors is not None else "gray")
    plt.ylabel(f"Average {metric}")
    plt.xlabel("Step number")
    plt.xticks(names, names)
    plt.title(title)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.legend(situations)
    if filename == None:
        plt.show()
    else:
        plt.savefig(f"../results/{filename}-{title}.png")

def compare_results_other_metrics(results, filename  = None, confidence=0.95, title="Agents Comparison", metric="", colors=None, plot=False):

    """Displays a bar plot comparing the performance of different agents/teams.

        Parameters
        ----------

        results: dict
            A dictionary where keys are the names and the values sequences of trials
        confidence: float
            The confidence level for the confidence interval
        title: str
            The title of the plot
        metric: str
            The name of the metric for comparison
        colors: Sequence[str]
            A sequence of colors (one for each agent/team)

        """
    situations = list(results.keys())
    results = list(results.values())
    x_pos = np.arange(len(results[0]))
    _, axs = plt.subplots(len(colors), 1, layout="constrained", figsize=(6 * len(colors), 8))
    # To allow iteration even with only one subplot
    if len(colors) == 1:
        axs = [axs]
    for pos in range(len(axs)):
        results_i = results[pos]
        names = [1] + list(range(5, len(results_i) + 1, len(results_i) // 20))
        means = [result.mean() for result in results_i]
        std_devs = [result.std() for result in results_i]
        N = [result.size for result in results_i]
        errors = [standard_error(std_devs[i], N[i], confidence) for i in range(len(means))]
        axs[pos].yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f} %')) # 1 decimal places
        axs[pos].set_ylabel(f"Average {metric}")
        axs[pos].set_xlabel("Step number")
        axs[pos].set_xticks(names)
        axs[pos].set_xticklabels(names)
        axs[pos].set_title(situations[pos])
        axs[pos].yaxis.grid(True)
        axs[pos].bar(x_pos, means, yerr=errors, align='center', alpha=0.5, color=colors[pos] if colors is not None else "gray", ecolor=colors[pos] if colors is not None else "gray", capsize=3)
    if filename == None:
        plt.show()
    else:
        plt.savefig(f"../results/{filename}-{title}.png")