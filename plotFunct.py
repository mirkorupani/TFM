import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from auxFunc import willmottSkillIndex, ksStatistic, pearsonCorrCoeff, perkinsSkillScore


def scatter_recon(model, reconstruction, title=None, returnMetrics=False, metricsToPlot=["mae", "bias", "willmott", "pearson", "perkin"]):
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

    metrics = {
        "mae": mean_absolute_error(model, reconstruction),
        "bias": np.mean(model - reconstruction),
        "willmott": willmottSkillIndex(model, reconstruction),
        "ks": ksStatistic(model, reconstruction),
        "pearson": pearsonCorrCoeff(model, reconstruction),
        "perkin": perkinsSkillScore(model, reconstruction)
    }
    text = ""
    for metric, value in metrics.items():
        if metric in metricsToPlot:
            text += f"{metric}: {value:.2g}\n"

    plt.text(0.95, 0.15, text, ha='right', va='center', transform=plt.gca().transAxes, fontsize=9)

    # Add title
    if title:
        plt.title(title)

    plt.legend(loc='upper left')

    if returnMetrics:
        return metrics


def combinedPlots(y, yPred, startIdx=0, title=None, savePath=None, waterlevel=False, returnMetrics=False):
    """Plot original data and predicted data, removing the first startIdx hours"""
    
    # Plot currents (and waterlevel)
    variables = ["u_x", "u_y"]
    metricsCurrents = individualCombinedPlot(y, yPred, variables, startIdx=startIdx, savePath=savePath, returnMetrics=returnMetrics, waterlevel=waterlevel, title=title)

    # Plot temperature and salinity (and waterlevel)
    variables = ["temperature", "salinity"]
    metricsTempSal = individualCombinedPlot(y, yPred, variables, startIdx=startIdx, savePath=savePath, returnMetrics=returnMetrics, waterlevel=waterlevel, title=title)
    
    if returnMetrics:
        return metricsCurrents | metricsTempSal


def individualCombinedPlot(y, yPred, variables, startIdx=0, savePath=None, returnMetrics=True, waterlevel=False, title=None):

    if returnMetrics:
        metrics = {}

    plt.figure(figsize=(18, 15) if waterlevel else (15, 10))
    
    if startIdx > 0:
        y = y.iloc[startIdx:]
        yPred = yPred.iloc[startIdx:]
    
    nCols = 3 if waterlevel else 2

    plt.subplot(nCols, 3, (1, 2))
    plt.plot(y.index, y[variables[0]], label="y")
    plt.plot(y.index, yPred[variables[0]], label="yPred")
    plt.title(f"Original and Predicted {variables[0]}")
    plt.legend()
    plt.subplot(nCols, 3, 3)
    if returnMetrics:
        metrics[variables[0]] = scatter_recon(y[variables[0]], yPred[variables[0]], title=variables[0], returnMetrics=returnMetrics)
    else:
        scatter_recon(y[variables[0]], yPred[variables[0]], title=variables[0])

    plt.subplot(nCols, 3, (4, 5))
    plt.plot(y.index, y[variables[1]], label="y")
    plt.plot(y.index, yPred[variables[1]], label="yPred")
    plt.title(f"Original and Predicted {variables[1]}")
    plt.legend()
    plt.subplot(nCols, 3, 6)
    if returnMetrics:
        metrics[variables[1]] = scatter_recon(y[variables[1]], yPred[variables[1]], title=variables[1], returnMetrics=returnMetrics)
    else:
        scatter_recon(y[variables[1]], yPred[variables[1]], title=variables[1], returnMetrics=returnMetrics)

    if waterlevel:
        plt.subplot(nCols, 3, (7, 8))
        plt.plot(y.index, y["waterlevel"], label="waterlevel")
        plt.plot(y.index, yPred["waterlevel"], label="waterlevelPred")
        plt.title("Original and Predicted waterlevel")
        plt.legend()
        plt.subplot(nCols, 3, 9)
        if returnMetrics:
            metrics["waterlevel"] = scatter_recon(y["waterlevel"], yPred["waterlevel"], title="waterlevel", returnMetrics=returnMetrics)
        else:
            scatter_recon(y["waterlevel"], yPred["waterlevel"], title="waterlevel")

    if title:
        plt.suptitle(title)
    
    if savePath:
        plt.savefig(f"{savePath}_{variables[0]}_{variables[1]}.png")
    
    plt.close()

    if returnMetrics:
        return metrics
