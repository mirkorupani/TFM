import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


def scatter_recon(model, reconstruction, title=None):
    plt.scatter(model, reconstruction, alpha=0.5)
    plt.xlabel('Model')
    plt.ylabel('Reconstruction')
    plt.plot([min(model), max(model)], [min(model), max(model)], color='r', linestyle='--', label='Perfect reconstruction')
    plt.gca().set_aspect('equal', adjustable='box')

    # Plot percentiles
    percentiles = np.percentile(model, [10, 20, 30, 40, 50, 60, 70, 80, 90])
    percentiles_pred = np.percentile(reconstruction, [10, 20, 30, 40, 50, 60, 70, 80, 90])
    plt.plot(percentiles, percentiles_pred, marker='D', color='black', linestyle='-', markersize=6, markerfacecolor='white', markeredgewidth=1.5, markeredgecolor='black', label='Percentiles')

    plt.xlim([np.min([np.percentile(model, 1), np.percentile(reconstruction, 1)]),
              np.max([np.percentile(model, 99), np.percentile(reconstruction, 99)])])
    plt.ylim(plt.gca().get_xlim())

    # Add MAE and bias with white alpha background
    mae = mean_absolute_error(model, reconstruction)
    bias = np.mean(model - reconstruction)
    plt.text(0.95, 0.05, f"MAE = {mae:.2g}\n bias = {bias:.2g}", ha='right', va='center', transform=plt.gca().transAxes)

    # Add title
    if title:
        plt.title(title)

    plt.legend(loc='upper left')


def combinedPlot(y, yPred, startIdx=0, title=None, savePath=None, waterlevel=False):
    """Plot original data and predicted data, removing the first startIdx hours"""
    
    plt.figure(figsize=(18, 15) if waterlevel else (15, 10))
    
    if startIdx > 0:
        y = y.iloc[startIdx:]
        yPred = yPred.iloc[startIdx:]
    
    nCols = 3 if waterlevel else 2

    plt.subplot(nCols, 3, (1, 2))
    plt.plot(y.index, y["u_x"], label="y")
    plt.plot(y.index, yPred["u_x"], label="yPred")
    plt.title("Original and Predicted u_x")
    plt.legend()
    plt.subplot(nCols, 3, 3)
    scatter_recon(y["u_x"], yPred["u_x"], title="u_x")

    plt.subplot(nCols, 3, (4, 5))
    plt.plot(y.index, y["u_y"], label="y")
    plt.plot(y.index, yPred["u_y"], label="yPred")
    plt.title("Original and Predicted u_y")
    plt.legend()
    plt.subplot(nCols, 3, 6)
    scatter_recon(y["u_y"], yPred["u_y"], title="u_y")

    if waterlevel:
        plt.subplot(nCols, 3, (7, 8))
        plt.plot(y.index, y["waterlevel"], label="waterlevel")
        plt.plot(y.index, yPred["waterlevel"], label="waterlevelPred")
        plt.title("Original and Predicted waterlevel")
        plt.legend()
        plt.subplot(nCols, 3, 9)
        scatter_recon(y["waterlevel"], yPred["waterlevel"], title="waterlevel")

    if title:
        plt.suptitle(title)
    
    if savePath:
        plt.savefig(savePath)
