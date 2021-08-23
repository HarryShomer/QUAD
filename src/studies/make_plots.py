import os
import json
import numpy as np 
import matplotlib.pyplot as plt

plt.style.use('ggplot')

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
IMG_DIR = os.path.join(CUR_DIR, "..", "..", "imgs")


def create_beta_plt(metric="MRR"):
    """
    Plot change in Beta hyperparameter vs. {metric} 
    """
    beta_vals = ["0", ".25", ".50", ".75", "1"]

    with open(os.path.join(CUR_DIR, "beta_study.json"), "r") as f:
        data = json.load(f)

    plt.figure()

    for dataset in data:
        metric_vals = [data[dataset][b] for b in beta_vals]   # Ensures in increasing order of Beta 
        # plt.plot(data['epoch'], data[metric])
        plt.plot(beta_vals, metric_vals, 'o-', label=dataset)

    plt.title(f'Impact of hyperparameter Beta on test {metric}')
    plt.xlabel('Value of Beta')
    plt.ylabel(f"Test {metric}")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(IMG_DIR, f"Beta_study_{metric}.png"))


def main():
    create_beta_plt()


if __name__ == "__main__":
    main()
