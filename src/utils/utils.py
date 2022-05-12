import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def plot_paper(
    sites=["nyu", "peking", "ohsu", "kki", "ni"],
    results=[0.1, 0.2, 0.3, 0.4, 0.5],
    path="test.png",
):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.style.use("classic")
    plt.margins(x=0.2)

    sites = list(map(lambda x: x.upper(), sites))
    results = np.array(results) * 100
    plt.plot(sites, results, "-o")
    plt.ylabel("Accuracy[%]")
    min_val = np.min(results)
    if min_val > 50:
        plt.ylim(50, 80)
    else:
        plt.ylim(min_val - 5, 80)

    # plt.savefig(path)
    return plt


def record_train_test(df_path: "Data/nitrc_niak/master_df.csv"):
    df = pd.read_csv(df_path)
    df["task"] = 0
    n_class = max(df.DX) + 1
    for site in df.Site.tolist():
        for c in range(n_class):
            sf = df[df.Site.isin([site])]
            sf = sf[sf.DX.isin([c])]
            train_index, test_index = train_test_split(
                sf.index, test_size=0.2, shuffle=True
            )
            df["task"][df.index.isin(train_index)] = "train"
            df["task"][df.index.isin(test_index)] = "test"
    df.to_csv(df_path)


if __name__ == "__main__":
    import numpy as np

    plot_paper(results=np.random.normal(0.65, 0.05, (5)))
