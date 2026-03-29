import argparse
import csv

import matplotlib.pyplot as plt
from matplotlib import rc_context
from utils import data_dir

reynolds_path = data_dir / "max_reynold.csv"
strouhal_path = data_dir / "strouhal.csv"


def Inch(d_cm: float) -> float:
    return d_cm / 2.54


rcCustom_wide = plt.rcParams.copy()
rcCustom_wide["figure.dpi"] = 150
rcCustom_wide["figure.figsize"] = (Inch(28.58), Inch(12.09))
rcCustom_wide["font.size"] = 10
rcCustom_wide["axes.titlesize"] = 14
rcCustom_wide["xtick.labelsize"] = 14
rcCustom_wide["ytick.labelsize"] = 14


@rc_context(rcCustom_wide)
def plot_reynolds() -> None:
    with reynolds_path.open("r") as f:
        reader = csv.reader(f)
        next(reader)
        labels, data = zip(*reader, strict=False)
        data = [float(x) for x in data]
        plt.bar(labels, data)
        plt.grid(axis="y")
        plt.title("Maximum Reynolds number")
        plt.savefig("reynolds.png")


@rc_context(rcCustom_wide)
def plot_strouhal() -> None:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("type", choices=["Reynolds", "Strouhal"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_type = args.type
    if plot_type == "Reynolds":
        plot_reynolds()
    elif plot_type == "Strouhal":
        plot_strouhal()


if __name__ == "__main__":
    main()
