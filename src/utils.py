import copy
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import os
import re

# Normalize dataset


def normalize_moments(dataset):
    # dataset is 1-d
    moments = np.zeros(2)
    moments[0] = dataset.mean()
    moments[1] = dataset.std()
    print("Moments:", moments[0], moments[1])
    return moments


def normalize_fit(dataset, moments):
    dataset = (dataset-moments[0])/moments[1]
    return dataset


def normalize_invert(dataset, moments):
    dataset = dataset*moments[1]+moments[0]
    return dataset


def min_max(dataset):
    # dataset is 1-d
    moments = np.zeros(2)
    moments[0] = dataset.min()
    moments[1] = dataset.max()
    print(moments[0], moments[1])
    return moments


def minmax_fit(dataset, moments):
    dataset = (dataset-moments[0])/(moments[1] - moments[0])
    return dataset


def minmax_invert(dataset, moments):
    dataset = dataset*(moments[1] - moments[0])+moments[0]
    return dataset

# convert an array of values (1-d time series) into a dataset matrix from


def create_dataset(dataset, look_back=1, predict_len=1):
    dataX, dataY = [], []
    for i in range(0, len(dataset)-look_back, predict_len):  # predict_len #look_back
        a = dataset[i:(i + look_back), 0]
        b = dataset[(i + predict_len):(i+look_back+predict_len), 0]
        dataX.append(a)
        dataY.append(b)
    return np.array(dataX), np.array(dataY)

# convert an array of values (1-d time series) into a dataset matrix from


def create_dataset2(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(0, dataset.shape[0]-look_back):  # predict_len #look_back
        a = dataset[i:(i + look_back), :]
        b = dataset[(i + 1):(i+look_back+1), :]
        dataX.append(a)
        dataY.append(b)
    return np.array(dataX), np.array(dataY)


def evaluation(predict, original):
    '''
    predict and original are 2-d vectors: [days, dimension]
    '''

    ae = np.abs(predict - original)
    se = ae**2
    ape = np.abs(ae/original)
    ape[ape == -np.inf] = np.NaN
    ape[ape == np.inf] = np.NaN

    mae = np.nanmean(ae)
    rmse = np.sqrt(np.nanmean(se))
    mape = np.nanmean(ape)

    print("rmse:", rmse, "mape:", mape)

    results = {"rmse": rmse, "mape": mape}

    return results


def PosteriorPlot(i, all_d_t_sampled, all_z_t_sampled, all_d_posterior, all_z_posterior_mean, plot_data, colormap, timestep, predict_dim, colorbar=True):

    # Transform the dataset to be the size of (batch,timestep*inputdim)
    plot_data = plot_data.transpose((1, 0, 2)).reshape(plot_data.shape[1], -1)

    # Make plot
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(20, 6))

    ax1.plot(np.arange((timestep)*predict_dim)/predict_dim+1+1/predict_dim/2, plot_data[i, :],
             color='#1f77b4', label="y")

    ax2 = ax1.twinx()
    ax2 = sns.heatmap(all_d_t_sampled[i, :, :].reshape(1, -1), linewidth=0.5,
                      cbar=colorbar, alpha=0.2, cmap=colormap, vmin=0, vmax=1)
    ax1.legend()

    ax3.plot(np.arange(len(all_z_t_sampled[i, :, :]))+1/2, all_z_t_sampled[i, :, :],
             alpha=1, color='#ff7f0e', label="Posterior sampled z")
    ax4 = ax3.twinx()
    ax4 = sns.heatmap(all_d_t_sampled[i, :, :].reshape(1, -1), linewidth=0.5, cbar=colorbar,
                      alpha=0.2, cmap=colormap, vmin=0, vmax=1)

    plt.suptitle("Posterior samples")
    plt.show()

    ############################
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(20, 6))

    ax1.plot(np.arange((timestep)*predict_dim)/predict_dim+1+1/predict_dim/2, plot_data[i, :],
             color='#1f77b4', label="y")
    ax2 = ax1.twinx()
    ax2 = sns.heatmap(all_d_posterior[i, :, 1].reshape(1, -1), linewidth=0.5, cbar=colorbar,
                      alpha=0.2, cmap=colormap, vmin=0, vmax=1)
    ax1.legend()

    ax3.plot(np.arange(len(all_z_posterior_mean[i, :, :]))+1/2, all_z_posterior_mean[i, :, :],
             alpha=1, color='#ff7f0e', label="Posterior mean of z")
    ax4 = ax3.twinx()
    ax4 = sns.heatmap(all_d_posterior[i, :, 1].reshape(1, -1), linewidth=0.5, cbar=colorbar,
                      alpha=0.2, cmap=colormap, vmin=0, vmax=1)

    plt.suptitle("Posterior mean")
    plt.show()

    print(all_d_t_sampled[i, :, :].reshape(-1))
    print(all_d_posterior[i, :, 1].reshape(-1))


def OriginalDataPlot(i, z_original, d_original, y_original2, colormap, timestep, predict_dim):
    fig, ax = plt.subplots(figsize=(20, 3))
    ax.plot(np.arange((timestep)*predict_dim)/predict_dim+1+1/predict_dim /
            2, y_original2[i:(i+timestep)], label="y", color="#1f77b4")
    ax.plot(np.arange(len(z_original[i:(i+timestep+1)]))+1/2,
            z_original[i:(i+timestep+1)], label="z", color='#ff7f0e')
    ax2 = ax.twinx()
    ax2 = sns.heatmap(d_original[i:(i+timestep+1)].reshape(1, -1),
                      linewidth=0.5, cbar=False, alpha=0.2, cmap=colormap, vmin=0, vmax=1)
    ax.legend()
    plt.suptitle("Original Data")


def PredictionPriorPlot(i, all_testForecast, all_testForecast_std, all_d_prior_sampled, all_z_prior_sampled, all_d_prior, all_z_prior_mean, testY, colormap, timestep, predict_dim, colorbar=True):

    y_original = testY[:, i, :]
    y_predict = all_testForecast[i]
    y_std = all_testForecast_std[i]

    d_t_sampled = all_d_prior_sampled[i, :]
    z_t_sampled = all_z_prior_sampled[i, :]

    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(20, 6))

    ax1.plot(np.arange(timestep*predict_dim)/predict_dim +
             1+1/predict_dim/2, y_original.reshape(-1, 1))
    ax1.plot(np.arange((timestep-1)*predict_dim, timestep*predict_dim) /
             predict_dim+1+1/predict_dim/2, y_predict.reshape(-1, 1))
    ax1.plot(np.arange((timestep-1)*predict_dim, timestep*predict_dim)/predict_dim +
             1+1/predict_dim/2, (y_predict-y_std).reshape(-1, 1), color='grey', alpha=0.5)
    ax1.plot(np.arange((timestep-1)*predict_dim, timestep*predict_dim)/predict_dim +
             1+1/predict_dim/2, (y_predict+y_std).reshape(-1, 1), color='grey', alpha=0.5)
    ax1.fill_between(np.arange((timestep-1)*predict_dim, timestep*predict_dim)/predict_dim+1+1/predict_dim /
                     2, (y_predict-y_std).reshape(-1), (y_predict+y_std).reshape(-1), color='grey', alpha=0.1)
    ax2 = ax1.twinx()
    ax2 = sns.heatmap(all_d_prior_sampled[i, :].reshape(
        1, -1), linewidth=0.5, cbar=colorbar, alpha=0.2, cmap=colormap, vmin=0, vmax=1)
    ax1.set_title(
        "Original Time Series and Prediction with Confidence Interval")

    ax3.plot(np.arange(len(all_z_prior_sampled[i, :]))+1/2, all_z_prior_sampled[i,
             :], alpha=1, label="Latent Factor z Samples", color='#ff7f0e')
    ax4 = ax3.twinx()
    ax4 = sns.heatmap(all_d_prior_sampled[i, :].reshape(
        1, -1), linewidth=0.5, cbar=colorbar, alpha=0.2, cmap=colormap, vmin=0, vmax=1)
    ax3.legend()

    fig.tight_layout()
    plt.show()

    ##########################

    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(20, 6))

    ax1.plot(np.arange(timestep*predict_dim)/predict_dim +
             1+1/predict_dim/2, y_original.reshape(-1, 1))
    ax1.plot(np.arange((timestep-1)*predict_dim, timestep*predict_dim) /
             predict_dim+1+1/predict_dim/2, y_predict.reshape(-1, 1))
    ax1.plot(np.arange((timestep-1)*predict_dim, timestep*predict_dim)/predict_dim +
             1+1/predict_dim/2, (y_predict-y_std).reshape(-1, 1), color='grey', alpha=0.5)
    ax1.plot(np.arange((timestep-1)*predict_dim, timestep*predict_dim)/predict_dim +
             1+1/predict_dim/2, (y_predict+y_std).reshape(-1, 1), color='grey', alpha=0.5)
    ax1.fill_between(np.arange((timestep-1)*predict_dim, timestep*predict_dim)/predict_dim+1+1/predict_dim /
                     2, (y_predict-y_std).reshape(-1), (y_predict+y_std).reshape(-1), color='grey', alpha=0.1)
    ax1.set_title(
        "Original Time Series and Prediction with Confidence Interval")
    ax2 = ax1.twinx()
    ax2 = sns.heatmap(all_d_prior_sampled[i, :].reshape(
        1, -1), linewidth=0.5, cbar=colorbar, alpha=0.2, cmap=colormap, vmin=0, vmax=1)

    ax3.plot(np.arange(len(all_z_prior_mean[i, :]))+1/2, all_z_prior_mean[i,
             :], alpha=1, label="Latent Factor z Prior mean", color='#ff7f0e')
    ax4 = ax3.twinx()
    ax4 = sns.heatmap(all_d_prior[i, :].reshape(
        1, -1), linewidth=0.5, cbar=colorbar, alpha=0.2, cmap=colormap, vmin=0, vmax=1)
    ax3.legend()

    fig.tight_layout()
    plt.show()

    print(all_d_prior_sampled[i, :].reshape(1, -1))
    print(all_d_prior[i, :].reshape(1, -1))


def PosteriorPredictionPlot(i, plot_data, all_d_t_sampled, all_z_t_sampled, all_d_posterior, all_z_posterior_mean, all_y_emission_mean, all_y_emission_std, predict_dim, timestep, forecaststep, colorbar=True, colormap="Blues"):

    y_original = plot_data[i+forecaststep, -(forecaststep)*predict_dim:]
    y_predict = all_y_emission_mean[i, timestep:timestep+forecaststep, :]
    y_std = all_y_emission_std[i, timestep:timestep+forecaststep, :]
    # Make plot
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(20, 6))

    ax1.plot(np.arange((timestep+forecaststep)*predict_dim)/predict_dim+1+1 /
             predict_dim/2, np.concatenate((plot_data[i, :], y_original)), color='#1f77b4')
    ax1.plot(np.arange((timestep)*predict_dim, (timestep+forecaststep)*predict_dim) /
             predict_dim+1+1/predict_dim/2, y_predict.reshape(-1, 1), color='#ff7f0e')
    ax1.plot(np.arange((timestep)*predict_dim, (timestep+forecaststep)*predict_dim) /
             predict_dim+1+1/predict_dim/2, (y_predict-y_std).reshape(-1, 1), color='grey', alpha=0.5)
    ax1.plot(np.arange((timestep)*predict_dim, (timestep+forecaststep)*predict_dim) /
             predict_dim+1+1/predict_dim/2, (y_predict+y_std).reshape(-1, 1), color='grey', alpha=0.5)
    ax1.fill_between(np.arange((timestep)*predict_dim, (timestep+forecaststep)*predict_dim)/predict_dim+1 +
                     1/predict_dim/2, (y_predict-y_std).reshape(-1), (y_predict+y_std).reshape(-1), color='grey', alpha=0.1)
    ax2 = ax1.twinx()
    ax2 = sns.heatmap(all_d_t_sampled[i, :].reshape(
        1, -1), linewidth=0.5, cbar=colorbar, alpha=0.2, cmap=colormap, vmin=0, vmax=1)
    ax2.vlines(timestep+1, 0, 1)
    ax1.set_title("Original Data")

    ax3.plot(np.arange(len(all_z_t_sampled[i, :]))+1/2,
             all_z_t_sampled[i, :], alpha=1, color='#2ca02c', label="Z Sampled")
    ax4 = ax3.twinx()
    ax4 = sns.heatmap(all_d_t_sampled[i, :].reshape(
        1, -1), linewidth=0.5, cbar=colorbar, alpha=0.2, cmap=colormap, vmin=0, vmax=1)
    ax4.vlines(timestep+1, 0, 1)
    ax4.set_title("Z Posterior|Z Prior")

    fig.tight_layout()
    plt.show()

    ############################

    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(20, 6))
    ax1.plot(np.arange((timestep+forecaststep)*predict_dim)/predict_dim+1+1 /
             predict_dim/2, np.concatenate((plot_data[i, :], y_original)), color='#1f77b4')
    ax1.plot(np.arange((timestep)*predict_dim, (timestep+forecaststep)*predict_dim) /
             predict_dim+1+1/predict_dim/2, y_predict.reshape(-1, 1), color='#ff7f0e')
    ax1.plot(np.arange((timestep)*predict_dim, (timestep+forecaststep)*predict_dim) /
             predict_dim+1+1/predict_dim/2, (y_predict-y_std).reshape(-1, 1), color='grey', alpha=0.5)
    ax1.plot(np.arange((timestep)*predict_dim, (timestep+forecaststep)*predict_dim) /
             predict_dim+1+1/predict_dim/2, (y_predict+y_std).reshape(-1, 1), color='grey', alpha=0.5)
    ax1.fill_between(np.arange((timestep)*predict_dim, (timestep+forecaststep)*predict_dim)/predict_dim+1 +
                     1/predict_dim/2, (y_predict-y_std).reshape(-1), (y_predict+y_std).reshape(-1), color='grey', alpha=0.1)
    ax2 = ax1.twinx()
    ax2 = sns.heatmap(all_d_posterior[i, :].reshape(
        1, -1), linewidth=0.5, cbar=colorbar, alpha=0.2, cmap=colormap, vmin=0, vmax=1)
    ax2.vlines(timestep+1, 0, 1)
    ax1.set_title("Original Data")

    ax3.plot(np.arange(len(all_z_posterior_mean[i, :]))+1/2,
             all_z_posterior_mean[i, :], alpha=1, color='#2ca02c')
    ax4 = ax3.twinx()
    ax4 = sns.heatmap(all_d_posterior[i, :].reshape(
        1, -1), linewidth=0.5, cbar=colorbar, alpha=0.2, cmap=colormap, vmin=0, vmax=1)
    ax4.vlines(timestep+1, 0, 1)
    ax4.set_title("Z Posterior|Z Prior")

    fig.tight_layout()
    plt.show()


def duration(status):
    Ss = copy.deepcopy(status)
    Ss[Ss == 0] = -1
    Ss.shape[0]
    D = 1
    D_list_1 = list()
    D_list_0 = list()

    for i in range(1, Ss.shape[0]):
        if Ss[i]*Ss[i-1] == 1:
            D = D+1
        if Ss[i]*Ss[i-1] == -1:
            if Ss[i-1] == 1:
                D_list_1.append(D)
            else:
                D_list_0.append(D)
            D = 1
    return np.array(D_list_1).mean(), np.array(D_list_0).mean()


def classification_scores(testy, yhat_classes):
    accuracy = accuracy_score(testy, yhat_classes)
    if accuracy < 0.5:
        yhat_classes = 1-yhat_classes
    accuracy = accuracy_score(testy, yhat_classes)
    f1 = f1_score(testy, yhat_classes)
    if f1 < 0.1:
        f1 = np.nan
    return accuracy, f1


def save_rmse_mape(dataname, res, type="Short-term", csv_filename="results/outputs/outputs_generated.csv"):
    """
    Save RMSE and MAPE results into a CSV file.

    Parameters:
        dataname (str): The name of the dataset.
        res (dict): A dictionary containing 'rmse' and 'mape'.
        csv_filename (str): The output CSV file path.
    """
    if dataname == "Sleep":
        dataname = "1 Sleep"
    elif dataname == "Unemployment":
        dataname = "2 Unemployment"
    elif dataname == "Hangzhou":
        if type == "Short-term":
            dataname = "3 Hangzhou"
        else:
            dataname = "1 Hangzhou"
    elif dataname == "Seattle":
        if type == "Short-term":
            dataname = "4 Seattle"
        else:
            dataname = "2 Seattle"
    elif dataname == "Pacific":
        if type == "Short-term":
            dataname = "5 Pacific"
        else:
            dataname = "3 Pacific"
    elif dataname == "Electricity":
        dataname = "6 Electricity"

    rmse = res.get("rmse")
    mape = res.get("mape")
    results = [
        {"Type": type, "Problem": dataname,
            "Metrics": "1 RMSE", "Method": "1 DS3M", "value": rmse},
        {"Type": type, "Problem": dataname,
            "Metrics": "2 MAPE (%)", "Method": "1 DS3M", "value": mape * 100}
    ]
    results_df = pd.DataFrame(results)

    if os.path.exists(csv_filename):
        results_df.to_csv(csv_filename, mode='a', header=False,
                          index=False, float_format="%.3f")
    else:
        results_df.to_csv(csv_filename, index=False, float_format="%.3f")

# %%


def save_results_to_csv(dataname, d_original, d_forecast, d_infer, res,
                        csv_filename="results/outputs/outputs_generated.csv", type_val=""):
    results = []

    # DS3M 指标
    acc_ds3m, f1_ds3m = classification_scores(d_original, d_forecast)

    # Inference 指标
    acc_infer, f1_infer = classification_scores(d_original, d_infer)

    if np.isnan(f1_ds3m) or np.isnan(f1_infer):
        return

    # Duration
    if dataname == 'Toy':
        dt1_ds3m, dt0_ds3m = duration(d_forecast)
        # if res.get("rmse") > 20:
        #     return
        results.append({"Type": type_val, "Problem": dataname,
                       "Metrics": "2 Duration for dt=1", "Method": "1 DS3M", "value": dt1_ds3m})
        results.append({"Type": type_val, "Problem": dataname,
                       "Metrics": "3 Duration for dt=0", "Method": "1 DS3M", "value": dt0_ds3m})
        results.append({"Type": type_val, "Problem": dataname,
                        "Metrics": "4 Accuracy (%)", "Method": "1 DS3M", "value": acc_ds3m})
        results.append({"Type": type_val, "Problem": dataname,
                        "Metrics": "5 F1 score", "Method": "1 DS3M", "value": f1_ds3m})
        results.append({"Type": type_val, "Problem": dataname,
                        "Metrics": "6 Inference - Accuracy (%)", "Method": "1 DS3M", "value": acc_infer})
        results.append({"Type": type_val, "Problem": dataname,
                        "Metrics": "7 Inference - F1 score", "Method": "1 DS3M", "value": f1_infer})
    elif dataname == "Lorenz":
        results.append({"Type": type_val, "Problem": dataname,
                        "Metrics": "2 Accuracy (%)", "Method": "1 DS3M", "value": acc_ds3m})
        results.append({"Type": type_val, "Problem": dataname,
                        "Metrics": "3 F1 score", "Method": "1 DS3M", "value": f1_ds3m})
        results.append({"Type": type_val, "Problem": dataname,
                        "Metrics": "4 Inference - Accuracy (%)", "Method": "1 DS3M", "value": acc_infer})
        results.append({"Type": type_val, "Problem": dataname,
                        "Metrics": "5 Inference - F1 score", "Method": "1 DS3M", "value": f1_infer})
    # 添加 rmse 行，新行的 Method 为 "RMSE"
    rmse = res.get("rmse")
    results.append({"Type": type_val, "Problem": dataname,
                   "Metrics": "1 RMSE", "Method": "1 DS3M", "value": rmse})

    results_df = pd.DataFrame(results)

    def extract_prefix(s):
        m = re.match(r"(\d+)", s)
        return int(m.group(1)) if m else float('inf')

    # 添加临时排序列，并排序
    results_df['MethodOrder'] = results_df['Method'].apply(extract_prefix)
    results_df['MetricOrder'] = results_df['Metrics'].apply(extract_prefix)
    results_df.sort_values(['MethodOrder', 'MetricOrder'], inplace=True)

    # 删除临时排序列，并调整列顺序
    results_df = results_df[['Type', 'Problem', 'Metrics', 'Method', 'value']]

    # 保存到 CSV 文件
    if os.path.exists(csv_filename):
        results_df.to_csv(csv_filename, mode='a', header=False,
                          index=False, float_format="%.3f")
    else:
        results_df.to_csv(csv_filename, index=False, float_format="%.3f")
