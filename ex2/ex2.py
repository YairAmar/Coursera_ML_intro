import numpy as np
from utils import load_data, plot_data
import sys


def main():
    x1, y1 = load_data(sys.argv[1])
    x2, y2 = load_data(sys.argv[2])
    plot_data(x1, y1)


if __name__ == "__main__":
    main()
