import matplotlib.pyplot as plt
import numpy as np

def plot_class_distribution(y, title="Class Distribution"):
    unique, counts = np.unique(y, return_counts=True)
    plt.bar(unique, counts)
    plt.xticks(unique, ["Class 0", "Class 1"])
    plt.title(title)
    plt.show()
