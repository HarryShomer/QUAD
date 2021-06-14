import os
import argparse
import numpy as np 
import matplotlib.pyplot as plt

plt.style.use('ggplot')

DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")


def get_data(file_name):
    """
    Retrieve Validation/Test data from a given file
    """
    data = {
        "epoch": [],
        "loss": [],
        "mrr": [],
        "hits@1": [],
        "hits@10": []
    }

    with open(os.path.join(DIR, file_name), "r") as f:
        for line in f:
            if "Vl_" in line:
                x = line.split()
                data['epoch'].append(int(x[1]) if x[1] != '0' else int(x[1][1:]))
                data["loss"].append(float(x[4]))
                data["hits@1"].append(float(x[7]))
                data["mrr"].append(float(x[10]))
                data["hits@10"].append(float(x[19]))
            
    return data


def create_epoch_plt(data, metric):
    """
    Plot Epoch vs Metric
    """
    plt.figure()

    plt.plot(data['epoch'], data[metric])

    plt.title(f'Epoch vs. {metric}')
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.tight_layout()

    plt.savefig(f"Epoch_vs_{metric}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="File to parse data from")
    args = parser.parse_args()

    data = get_data(args.file)

    for m in ['mrr', 'hits@1', 'hits@10']:
        create_epoch_plt(data, m)



if __name__ == "__main__":
    main()
