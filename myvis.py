import matplotlib.pyplot as plt


def set_colormap(cmap="Set2"):
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.get_cmap(cmap).colors)
