import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.style.use('classic')


def plot_paper(
    sites=["nyu", "peking", "ohsu", "kki", "ni"],
    results=[0.1, 0.2, 0.3, 0.4, 0.5],
    path="test.png",
):
    sites = list(map(lambda x: x.upper(), sites))
    results = np.array(results) * 100
    plt.plot(sites, results, "-o")
    plt.ylabel("Accuracy[%]")
    plt.ylim(50, 80)
    plt.savefig(path)


if __name__ == "__main__":
    import numpy as np

    plot_paper(results=np.random.normal(0.65, 0.05, (5)))
