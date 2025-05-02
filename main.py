# %%
import warnings
import os
import copy
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import matplotlib
import argparse
from src.DSSSMCode import *
from src.utils import *
import torch.utils.data
import torch.utils
import torch
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
warnings.filterwarnings("ignore")

# %%
remove_mean = False
remove_residual = False
longterm = False
bidirection = False

# %%
parser = argparse.ArgumentParser(description='DS3M')
parser.add_argument(
    "-f", "--fff", help="a dummy argument to fool ipython", default="1")
parser.add_argument('-p', '--problem', type=str,
                    help='problem', default='Unemployment')
parser.add_argument('-train', '--train', action='store_true',
                    help='whether to retrain the model')
parser.add_argument('--seed', type=int, default=1)

# cmd = ['-p', "Toy"]
# cmd = ['-p', "Lorenz"]
# cmd = ['-p', "Sleep"]
# cmd = ['-p', "Unemployment"]
# cmd = ['-p', "Hangzhou"]
# cmd = ['-p', "Seattle", '--train']
# cmd = ['-p', "Pacific"]
# cmd = ['-p', "Electricity"]
# args = parser.parse_args(cmd)
args = parser.parse_args()
print(args)

if args.seed is not None:
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# %%
dataname = args.problem
print(dataname)
restore = not args.train
retry = 3

if dataname == 'Toy':

    freq = 1
    test_len = 500
    timestep = 20
    predict_dim = 1

    x_dim = predict_dim  # Dimension of x
    y_dim = predict_dim  # Dimension of y #equal to predict_len
    h_dim = 10  # Dimension of the hidden states in RNN
    z_dim = 2  # Dimension of the latent variable z
    d_dim = 2  # Dimension of the latent variable d
    n_layers = 1  # Number of the layers of the RNN
    clip = 10  # Gradient clips
    learning_rate = 1e-3  # Learning rate
    batch_size = 64  # Batch size
    n_epochs = 100  # Number of epochs for training

    RawDataOriginal = pd.read_csv(
        'data/Toy/simulation_data_nonlinear_y.csv', header=None).values
    RawDataOriginal = RawDataOriginal.reshape(-1, 1, 1)

# %%
if dataname == 'Lorenz':

    freq = 1
    test_len = 1000
    timestep = 5
    predict_dim = 10
    look_back = timestep

    # hyperparameters
    x_dim = predict_dim  # Dimension of x
    y_dim = predict_dim  # Dimension of y #equal to predict_len
    h_dim = 20  # Dimension of the hidden states in RNN
    z_dim = 3  # Dimension of the latent variable z
    d_dim = 2  # Dimension of the latent variable d
    n_layers = 1  # Number of the layers of the RNN
    clip = 10  # Gradient clips
    learning_rate = 1e-3  # Learning rate
    batch_size = 64  # Batch size
    n_epochs = 300

    import json
    f = open("data/Lorenz/lorenz.json")
    data_st_all = json.load(f)
    D = len(data_st_all['data'][0][0])
    factors_true = torch.FloatTensor(data_st_all['factors'])
    z_true = torch.FloatTensor(data_st_all['latents'])
    z_true = z_true[:, 2000:5000]
    data_st = np.array(data_st_all['data'])
    data_st = data_st[:, 2000:5000]
    data_st = data_st + \
        data_st[0].std(axis=0)*0.001 * \
        np.random.randn(data_st.shape[1], 10)  # added noise
    data_st = (data_st-data_st[0].mean(axis=0))/data_st[0].std(axis=0)  # added
    states = np.zeros(z_true.numpy().shape[0:2])
    states[z_true.numpy()[:, :, 0] > 0] = 1
    states = torch.LongTensor(states)
    RawDataOriginal = data_st.transpose(1, 0, 2)

# %%
if dataname == 'Sleep':

    freq = 1
    timestep = 200
    predict_dim = 1
    test_len = 1000 - timestep

    x_dim = predict_dim  # Dimension of x
    y_dim = predict_dim  # Dimension of y #equal to predict_len
    h_dim = 10  # Dimension of the hidden states in RNN
    z_dim = 1  # Dimension of the latent variable z
    d_dim = 2  # Dimension of the latent variable d
    n_layers = 1  # Number of the layers of the RNN
    clip = 10  # Gradient clips
    learning_rate = 1e-3  # Learning rate
    batch_size = 64  # Batch size
    n_epochs = 100

    dataset = pd.read_csv("data/Sleep/b1.txt", sep=" ", header=None)
    chest = dataset.iloc[:, 1].values.reshape(-1, 1)
    train_data = chest[6200:7200, :]
    test_data = chest[5200:6200, :]
    RawDataOriginal = np.concatenate((train_data, test_data)).reshape(-1, 1, 1)

# %%
if dataname == 'Unemployment':
    freq = 1
    test_len = 20*12
    timestep = 6
    predict_dim = 1

    x_dim = predict_dim  # Dimension of x
    y_dim = predict_dim  # Dimension of y #equal to predict_len
    h_dim = 10  # Dimension of the hidden states in RNN
    z_dim = 2  # Dimension of the latent variable z
    d_dim = 2  # Dimension of the latent variable d
    n_layers = 1  # Number of the layers of the RNN
    clip = 10  # Gradient clips
    learning_rate = 1e-3  # Learning rate
    batch_size = 64  # Batch size
    n_epochs = 500  # Number of epochs for training

    RawDataOriginal = pd.read_csv(
        'data/Unemployment/UNRATE.csv', header=0).loc[:, 'UNRATE'].values
    RawDataOriginal = RawDataOriginal.reshape(-1, 1, 1)
    torch.manual_seed(19)

# %%
if dataname == "Hangzhou":

    remove_mean = True
    bidirection = True

    freq = 108
    test_len = 5 * freq
    timestep = 12  # freq #freq*1
    predict_dim = 80

    # hyperparameters
    x_dim = predict_dim  # Dimension of x
    y_dim = predict_dim  # Dimension of y #equal to predict_len
    h_dim = predict_dim  # Dimension of the hidden states in RNN
    z_dim = 40  # int(predict_dim/2) # Dimension of the latent variable z
    d_dim = 2  # Dimension of the latent variable d
    n_layers = 1  # Number of the layers of the RNN
    clip = 10  # Gradient clips
    learning_rate = 1e-3  # Learning rate
    batch_size = 64  # 512 # Batch size
    n_epochs = 100

    from scipy.io import loadmat
    RawDataOriginal = loadmat(
        'data/Hangzhou/hangzhou.mat')['tensor'].astype('float')
    RawDataOriginal = RawDataOriginal.transpose(1, 2, 0)
    RawDataOriginal.shape

# %%
if dataname == "Seattle":
    remove_residual = True
    bidirection = True

    freq = 288
    test_len = 5 * freq
    timestep = 12  # 24 #int(freq/3)
    predict_dim = 323

    # hyperparameters
    x_dim = predict_dim  # Dimension of x
    y_dim = predict_dim  # Dimension of y #equal to predict_len
    h_dim = 40  # Dimension of the hidden states in RNN
    z_dim = 10  # Dimension of the latent variable z
    d_dim = 2  # Dimension of the latent variable d
    n_layers = 1  # Number of the layers of the RNN
    clip = 10  # Gradient clips
    learning_rate = 1e-2  # Learning rate
    batch_size = 64  # Batch size
    n_epochs = 100

    # Dataset Preprocessing
    RawDataOriginal = np.load(
        "data/Seattle/seattle.npz")['arr_0'].astype('float').transpose(1, 0)
    RawDataOriginal = RawDataOriginal.reshape(
        (-1, freq, RawDataOriginal.shape[1]))
    RawDataOriginal.shape

# %%
if dataname == "Pacific":

    remove_mean = True
    bidirection = True

    freq = 12
    test_len = 5 * freq
    timestep = 24  # 12#freq*3
    predict_dim = 2520

    # hyperparameters
    x_dim = predict_dim  # Dimension of x
    y_dim = predict_dim  # Dimension of y #equal to predict_len
    h_dim = 200  # Dimension of the hidden states in RNN
    z_dim = 50  # Dimension of the latent variable z
    d_dim = 2  # Dimension of the latent variable d
    n_layers = 1  # Number of the layers of the RNN
    clip = 10  # Gradient clips
    learning_rate = 1e-3  # Learning rate
    batch_size = 64  # Batch size
    n_epochs = 100

    # Dataset Preprocessing
    RawDataOriginal = pd.read_csv(
        'data/Pacific/pacific.tsv', sep='\t', header=None).values.reshape(-1, 30*84)[3:]
    RawDataOriginal = RawDataOriginal.reshape(-1, 12, RawDataOriginal.shape[1])
    RawDataOriginal.shape

# %%
if dataname == 'Electricity':

    freq = 1
    timestep = 14
    predict_dim = 48
    test_len = 320
    DataPath = 'data/Electricity/French_all.csv'

    x_dim = predict_dim  # Dimension of x
    y_dim = predict_dim  # Dimension of y #equal to predict_len
    h_dim = predict_dim  # Dimension of the hidden states in RNN
    z_dim = 10  # Dimension of the latent variable z
    d_dim = 2  # Dimension of the latent variable d
    n_layers = 1  # Number of the layers of the RNN
    clip = 10  # Gradient clips
    learning_rate = 1e-3  # Learning rate
    batch_size = 64  # 256 # Batch size
    n_epochs = 100  # Number of epochs for training

    RawDataOriginal = pd.read_csv(DataPath)
    RawDataOriginal = RawDataOriginal[RawDataOriginal["Date"] < '2019-12-31']
    RawDataOriginal = RawDataOriginal['Load'].values
    RawDataOriginal = RawDataOriginal.reshape(-1, 1, predict_dim)

# %%
if remove_mean:
    means = np.expand_dims(
        np.mean(RawDataOriginal[:-int(test_len/freq), :, :], axis=0), axis=0)
else:
    means = 0
RawData = RawDataOriginal - means

RawData = RawData.reshape(-1, RawData.shape[2])
data = RawData

if remove_residual:
    trend = data[0:-1, :]
    data = data[1:, :] - trend

# Split into train and test data
if dataname == 'Unemployment':
    length = len(data) - test_len
    train_len = int(length)
    train_data = data[:train_len]
    valid_data = data[:train_len]
    test_data = data[(-test_len-timestep-1):-1]

elif dataname == 'Lorenz':
    train_len = 1000 + timestep
    valid_len = len(data) - train_len - test_len
    train_data = data[:train_len]
    valid_data = data[(train_len):(train_len+valid_len)]
    test_data = data[(-test_len-timestep-1):-1]

elif dataname == 'Sleep':
    train_len = 1000
    train_data = data[:train_len]
    valid_data = data[:train_len]
    test_data = data[-test_len-timestep:]

    print("train size (days):", train_data.shape,
          "valid size(days):", valid_data.shape,
          "test size(days):", test_data.shape)

else:
    length = len(data) - test_len
    train_len = int(length * 0.75)
    valid_len = int(length * 0.25)

    train_data = data[:train_len]
    valid_data = data[(train_len):(train_len+valid_len)]
    test_data = data[(-test_len-timestep-1):-1]

print("train size (days):", train_data.shape[0],
      "valid size(days):", valid_data.shape[0],
      "test size(days):", test_data.shape[0])


# %%
# Normalize the dataset
moments = normalize_moments(train_data)
train_data = normalize_fit(train_data, moments)
valid_data = normalize_fit(valid_data, moments)
test_data = normalize_fit(test_data, moments)

# Create training and test dataset
trainX, trainY = create_dataset2(train_data, timestep)
validX, validY = create_dataset2(valid_data, timestep)
testX, testY = create_dataset2(test_data, timestep)

trainX = np.transpose(trainX, (1, 0, 2))
validX = np.transpose(validX, (1, 0, 2))
testX = np.transpose(testX, (1, 0, 2))
print("3D size(X):", trainX.shape, validX.shape, testX.shape)

trainY = np.transpose(trainY, (1, 0, 2))
validY = np.transpose(validY, (1, 0, 2))
testY = np.transpose(testY, (1, 0, 2))
print("3D size(Y):", trainY.shape, validY.shape, testY.shape)

print("Numpy into Tensor, done!")
trainX = torch.from_numpy(trainX).float()
validX = torch.from_numpy(validX).float()
testX = torch.from_numpy(testX).float()
trainY = torch.from_numpy(trainY).float()
validY = torch.from_numpy(validY).float()
testY = torch.from_numpy(testY).float()

# %%
directoryBest = os.path.join("results", "checkpoints", dataname)
figdirectory = os.path.join("figures")
if not os.path.exists(directoryBest):
    os.makedirs(directoryBest)
if not os.path.exists(figdirectory):
    os.makedirs(figdirectory)
figdirectory = figdirectory+'/' + dataname + '_'


# %%
# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainX = trainX.to(device)
validX = validX.to(device)
testX = testX.to(device)
trainY = trainY.to(device)
validY = validY.to(device)
testY = testY.to(device)

# %%
# Training
start = time.time()

def forecast(model, testX, testY, forecaststep=1, MC_S=200):

    forecast_MC, forecast_d_MC, forecast_z_MC = model._forecastingMultiStep(
        testX, testY, forecaststep, MC_S)

    # print(forecast_MC.shape, forecast_d_MC.shape, forecast_z_MC.shape)

    if forecaststep == 1:
        all_testForecast = normalize_invert(
            forecast_MC.squeeze(1).transpose(1, 0, 2), moments)
    else:
        all_testForecast = normalize_invert(
            forecast_MC.squeeze(2).transpose(1, 0, 2), moments)

    testY_inversed = normalize_invert(
        testY.cpu().numpy().transpose(1, 0, 2), moments)
    size = testY_inversed.shape[0]

    forecast_d_MC_argmax = []
    for i in range(d_dim):
        forecast_d_MC_argmax.append(
            np.sum(forecast_d_MC[:, -1, :, :] == i, axis=0))
    forecast_d_MC_argmax = np.argmax(
        np.array(forecast_d_MC_argmax), axis=0).reshape(-1)

    if remove_mean:
        testForecast_mean = np.mean(
            all_testForecast, axis=1) + np.tile(means[0, :, :], (int(test_len/freq), 1))
        testForecast_uq = np.quantile(
            all_testForecast, 0.95, axis=1) + np.tile(means[0, :, :], (int(test_len/freq), 1))
        testForecast_lq = np.quantile(
            all_testForecast, 0.05, axis=1) + np.tile(means[0, :, :], (int(test_len/freq), 1))
    elif remove_residual:
        testForecast_mean = np.mean(
            all_testForecast, axis=1) + trend[-test_len:, :]
        testForecast_uq = np.quantile(
            all_testForecast, 0.95, axis=1) + trend[-test_len:, :]
        testForecast_lq = np.quantile(
            all_testForecast, 0.05, axis=1) + trend[-test_len:, :]
    else:
        testForecast_mean = np.mean(all_testForecast, axis=1)
        testForecast_uq = np.quantile(all_testForecast, 0.95, axis=1)
        testForecast_lq = np.quantile(all_testForecast, 0.05, axis=1)

    testOriginal = RawDataOriginal[-int(test_len/freq):, :, :].reshape(-1, RawDataOriginal.shape[2])
    # print(testForecast_mean.shape, testOriginal.shape)

    # Evaluation results
    res = evaluation(testForecast_mean.T, testOriginal.T)

    return res, testForecast_mean, testOriginal, size, forecast_d_MC_argmax, testForecast_uq, testForecast_lq


# %%
if restore == False:

    unique_states = 0

    while unique_states < 2:
        # Init model
        model = DSSSM(x_dim, y_dim, h_dim, z_dim, d_dim,n_layers, device, bidirection).to(device)
        # Optimizer
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("The total number of parameters:", total_params)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        early_stopping = EarlyStopping(20, verbose=True)
        loss_train_list, loss_valid_list, loss_test_list = [], [], []
        best_validation = 1e5
        best_validation_temp = 1e5
        
        for i in range(retry):
            print("n_epochs", n_epochs)
            for epoch in range(1, n_epochs + 1):

                # Training
                all_d_t_sampled_train, all_z_t_sampled_train, loss_train, all_d_posterior_train, all_z_posterior_mean_train = train(
                    model, optimizer, trainX, trainY, epoch, batch_size, n_epochs)

                # Validation
                if dataname in ['Unemployment', 'Sleep']:
                    loss_valid = loss_train
                else:
                    all_d_t_sampled_valid, all_z_t_sampled_valid, loss_valid, all_d_posterior_valid, all_z_posterior_mean_valid = test(
                        model, validX, validY, epoch, "valid")

                # Testing
                all_d_t_sampled_test, all_z_t_sampled_test, loss_test, all_d_posterior_test, all_z_posterior_mean_test = test(
                    model, testX, testY, epoch, "test")
                loss_train_list.append(loss_train)
                loss_valid_list.append(loss_valid)
                loss_test_list.append(loss_test)

                if (loss_valid < best_validation):
                    best_validation = copy.deepcopy(loss_valid)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss_train,
                    }, os.path.join(directoryBest, 'best_temp.tar'))

                # Learning rate scheduler
                scheduler.step(loss_valid)
                print("Learning rate:", optimizer.param_groups[0]['lr'])

                # Early stopping
                loss_valid_average = np.average(loss_valid_list)
                early_stopping(loss_valid_average, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            print("Running Time:", time.time()-start)

            if best_validation < best_validation_temp:
                best_validation_temp = best_validation
                PATH = os.path.join(directoryBest, 'best.tar')
                checkpoint = torch.load(os.path.join(
                    directoryBest, 'best_temp.tar'))
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                torch.save({
                    'epoch': checkpoint['epoch'],
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': checkpoint['loss'],
                }, PATH)

        #训练后检查离散状态数
        _, _, _, _, forecast_d_MC_argmax, _, _ = forecast(model, testX, testY, forecaststep=1, MC_S=200)
        unique_states = len(np.unique(forecast_d_MC_argmax))

    os.remove(os.path.join(directoryBest, 'best_temp.tar'))


# %%
# Reload the parameters
if restore:
    PATH = os.path.join(directoryBest, 'checkpoint.tar')
else:
    PATH = os.path.join(directoryBest, 'best.tar')

model = DSSSM(x_dim, y_dim, h_dim, z_dim, d_dim,
              n_layers, device, bidirection).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

epoch = checkpoint['epoch']
loss = checkpoint['loss']
print("Epoch:", epoch)
total_params = sum(p.numel() for p in model.parameters())
# print("The total number of parameters:", total_params)


# %%
res, testForecast_mean, testOriginal, size, forecast_d_MC_argmax, testForecast_uq, testForecast_lq = forecast(
    model, testX, testY, forecaststep=1, MC_S=200)

my_cmap = matplotlib.cm.get_cmap('rainbow')
cmap = plt.get_cmap('RdBu', d_dim)

# %%
if dataname == 'Toy':

    d_original = pd.read_csv('data/Toy/simulation_data_nonlinear_d.csv',
                             header=None).values.reshape(-1)[-test_len:]
    z_original = pd.read_csv('data/Toy/simulation_data_nonlinear_z.csv',
                             header=None).values.reshape(-1)[-test_len:]

    DSARF = pd.read_csv(
        "data/Toy/Toy_s_forecasted_dsarf.csv", header=None).values
    DSARF = np.concatenate((np.array([[1]]), DSARF))
    SNLDS = pd.read_csv(
        "data/Toy/Toy_s_forecasted_snlds.csv", header=None).values

    DS3M = forecast_d_MC_argmax  # d(state) value
    acc = accuracy_score(d_original[-size:], DS3M)
    if acc < 0.5:
        DS3M = 1-DS3M
    DSARF = DSARF
    SNLDS = 1-SNLDS[-size:]

    inferX = torch.from_numpy(np.expand_dims(
        data[(-test_len-1):-1], axis=0)).float().to(device)
    inferY = torch.from_numpy(np.expand_dims(
        data[-test_len:], axis=0)).float().to(device)
    all_d_t_sampled_plot_test, all_z_t_sampled_test, loss_test, all_d_posterior_test, all_z_posterior_mean_test = test(
        model, inferX, inferY, 0, "test")
    d_infer = all_d_t_sampled_plot_test[:, 1, 0]
    d_infer = d_infer

    xticks_int = 10
    fig, (ax1, ax4, ax3, ax5, ax6) = plt.subplots(5, 1, figsize=(20, 10),
                                                  sharex=True, gridspec_kw={'height_ratios': [3, 0.5, 0.5, 0.5, 0.5]})

    ######
    _ = ax1.plot(np.arange(size*predict_dim)/predict_dim, testOriginal.reshape(-1),
                 label='original $y_t$')
    _ = ax1.plot(np.arange(size*predict_dim)/predict_dim, testForecast_mean.reshape(-1),
                 color='red', label='forecasted mean for $y_t$')
    _ = ax1.plot(np.arange(size*predict_dim)/predict_dim,
                 testForecast_uq.reshape(-1), color='grey', alpha=0.8)
    _ = ax1.plot(np.arange(size*predict_dim)/predict_dim,
                 testForecast_lq.reshape(-1), color='grey', alpha=0.8)
    _ = ax1.fill_between(np.arange(size*predict_dim)/predict_dim,
                         testForecast_uq.reshape(-1),
                         testForecast_lq.reshape(-1), color='grey', alpha=0.2, label='90% confidence interval')
    _ = ax1.set_xlim(0, size)
    _ = ax1.legend()
    _ = ax1.set_title("Prediction for $y_t$")

    ######
    _ = sns.heatmap(d_original[-size:].reshape(1, -1), linewidth=0,
                    cbar=False, alpha=1, cmap=cmap, vmin=0, vmax=1, ax=ax4)
    _ = ax4.set_xticks(np.round(np.arange(0, size, xticks_int)))
    _ = ax4.set_xticklabels(np.round(np.arange(0, size, xticks_int)))
    _ = ax4.set_ylabel("True", fontweight='bold', c='red')
    _ = ax4.set_yticks([])

    ######
    _ = sns.heatmap(DS3M.reshape(1, -1), linewidth=0, cbar=False,
                    alpha=1, cmap=cmap, vmin=0, vmax=1, ax=ax3)
    _ = ax3.set_xticks(np.round(np.arange(0, size, xticks_int)))
    _ = ax3.set_xticklabels(np.round(np.arange(0, size, xticks_int)))
    _ = ax3.set_ylabel("DS$^3$M", fontweight='bold')
    _ = ax3.set_yticks([])

    _ = sns.heatmap(DSARF.reshape(1, -1), linewidth=0, cbar=False,
                    alpha=1, cmap=cmap, vmin=0, vmax=1, ax=ax5)
    _ = ax5.set_xticks(np.round(np.arange(0, size, xticks_int)))
    _ = ax5.set_xticklabels(np.round(np.arange(0, size, xticks_int)))
    _ = ax5.set_ylabel("DSARF", fontweight='bold')
    _ = ax5.set_yticks([])

    _ = sns.heatmap(SNLDS.reshape(1, -1), linewidth=0, cbar=False,
                    alpha=1, cmap=cmap, vmin=0, vmax=1, ax=ax6)
    _ = ax6.set_xticks(np.round(np.arange(0, size, xticks_int)))
    _ = ax6.set_xticklabels(np.round(np.arange(0, size, xticks_int)))
    _ = ax6.set_ylabel("SNLDS", fontweight='bold')
    _ = ax6.set_yticks([])
    plt.savefig(figdirectory+"Prediction.png", format='png')
    plt.show()

    save_results_to_csv(
        dataname, d_original[-size:], DS3M, d_infer, res, type_val="")

# %%
if dataname == 'Lorenz':

    trainX2 = torch.from_numpy(RawDataOriginal[:1000, :, :]).float().to(device)
    trainY2 = torch.from_numpy(
        RawDataOriginal[1:1001, :, :]).float().to(device)
    all_d_t_sampled_plot_test, all_z_t_sampled_test, loss_test, all_d_posterior_test, all_z_posterior_mean_test = test(
        model, trainX2, trainY2, 0, "test")
    latents = z_true[0, 1:1001, :]
    categories = all_d_t_sampled_plot_test[:, 1:, :].reshape(-1)
    colormap = np.array(['r', 'b', 'g', 'y'])
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(latents[:, 0], latents[:, 1],
               latents[:, 2], c=colormap[categories])
    plt.savefig(figdirectory+"Prediction.png", format='png')
    plt.show()

    testX2 = torch.from_numpy(
        RawDataOriginal[(-test_len-1):-1, :, :]).float().to(device)
    testY2 = torch.from_numpy(
        RawDataOriginal[-test_len:, :, :]).float().to(device)
    all_d_t_sampled_plot_test, all_z_t_sampled_test, loss_test, all_d_posterior_test, all_z_posterior_mean_test = test(
        model, testX2, testY2, 0, "test")
    categories = all_d_t_sampled_plot_test[:, 1:, :].reshape(-1)

    save_results_to_csv(dataname,
                        states.numpy().reshape(-1)[-testX.shape[1]:],
                        forecast_d_MC_argmax,
                        categories,
                        res,
                        type_val="")

# %%
if dataname == 'Sleep':

    xticks_int = 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 4),
                                   sharex=True, gridspec_kw={'height_ratios': [3, 0.5]})
    ax1.plot(np.arange(size*predict_dim)/predict_dim,
             testOriginal.reshape(-1), label='original')
    ax1.plot(np.arange(size*predict_dim)/predict_dim,
             testForecast_mean.reshape(-1), color='red', label='forecasted mean')
    ax1.plot(np.arange(size*predict_dim)/predict_dim,
             testForecast_uq.reshape(-1), color='grey', alpha=0.8)
    ax1.plot(np.arange(size*predict_dim)/predict_dim,
             testForecast_lq.reshape(-1), color='grey', alpha=0.8)
    ax1.fill_between(np.arange(size*predict_dim)/predict_dim,
                     testForecast_uq.reshape(-1),
                     testForecast_lq.reshape(-1), color='grey', alpha=0.2)
    ax1.legend()

    sns.heatmap(1-forecast_d_MC_argmax.reshape(1, -1), linewidth=0,
                cbar=False, alpha=1, cmap=cmap, vmin=0, vmax=1, ax=ax2)
    ax2.set_xticks(np.round(np.arange(0, size, xticks_int)))
    ax2.set_xticklabels(np.round(np.arange(0, size, xticks_int)))
    ax5 = ax2.twinx()

    plt.savefig(figdirectory + 'Prediction.png', format='png')
    plt.show()

    save_rmse_mape(dataname, res)

# %%
if dataname == 'Unemployment':

    xticklabels = list()
    for i in np.arange(2002, 2022):
        xticklabels.append(str(i) + " Jan")

    xticks_int = 100
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 4),
                                   sharex=True, gridspec_kw={'height_ratios': [3, 0.5]})

    _ = ax1.plot(testOriginal.reshape(-1), label='original')
    _ = ax1.plot(testForecast_mean.reshape(-1), color='red', label='predict')
    _ = ax1.plot(testForecast_uq.reshape(-1), color='grey', alpha=0.8)
    _ = ax1.plot(testForecast_lq.reshape(-1), color='grey', alpha=0.8)
    _ = ax1.fill_between(np.arange(size), testForecast_uq.reshape(-1),
                         testForecast_lq.reshape(-1), color='grey', alpha=0.4)
    _ = ax1.legend()

    _ = sns.heatmap(1-forecast_d_MC_argmax.reshape(1, -1), linewidth=0,
                    cbar=False, alpha=1, cmap=cmap, vmin=0, vmax=1, ax=ax2)

    _ = ax2.set_xticks(np.arange(9, test_len, 12))
    _ = ax2.set_xticklabels(xticklabels)
    _ = plt.xticks(rotation=0)
    _ = plt.suptitle('%s' % (dataname))
    plt.savefig(figdirectory + 'Prediction.png', format='png')
    plt.show()

    save_rmse_mape(dataname, res)

# %%
if dataname == 'Hangzhou':
    for station in [0, 40]:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4),
                                       sharex=True, gridspec_kw={'height_ratios': [3, 0.5]})
        ax1.plot(testOriginal[:, station], label='original')
        ax1.plot(testForecast_mean[:, station], label='predict')
        ax1.plot(testForecast_uq[:, station], color='grey', alpha=0.8)
        ax1.plot(testForecast_lq[:, station], color='grey', alpha=0.8)
        ax1.fill_between(np.arange(
            size), testForecast_uq[:, station], testForecast_lq[:, station], color='grey', alpha=0.4)
        ax1.legend()
        sns.heatmap(1-forecast_d_MC_argmax.reshape(1, -1), linewidth=0,
                    cbar=False, alpha=1, cmap=cmap, vmin=0, vmax=1, ax=ax2)

        plt.suptitle('%s #%i' % (dataname, station))
        plt.savefig(figdirectory+'Station %i' % (station)+'.png', format='png')
        plt.show()

    save_rmse_mape(dataname, res)

    print("Long term:")
    res, testForecast_mean, testOriginal, size, forecast_d_MC_argmax, testForecast_uq, testForecast_lq = forecast(
        model, testX[:, 0:1, :], testY[:, 0:1, :], test_len, MC_S=200)

    save_rmse_mape(dataname, res, "Long-term")

# %%
if dataname == 'Seattle':
    for station in [0, 322]:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4),
                                       sharex=True, gridspec_kw={'height_ratios': [3, 0.5]})
        ax1.plot(testOriginal[:, station], label='original')
        ax1.plot(testForecast_mean[:, station], label='predict')
        ax1.plot(testForecast_uq[:, station], color='grey', alpha=0.8)
        ax1.plot(testForecast_lq[:, station], color='grey', alpha=0.8)
        ax1.fill_between(np.arange(
            size), testForecast_uq[:, station], testForecast_lq[:, station], color='grey', alpha=0.4)
        ax1.legend()
        sns.heatmap(forecast_d_MC_argmax.reshape(1, -1), linewidth=0,
                    cbar=False, alpha=1, cmap=cmap, vmin=0, vmax=1, ax=ax2)

        plt.suptitle('%s #%i' % (dataname, station))
        plt.savefig(figdirectory+'Station %i' % (station)+'.png', format='png')
        plt.show()

    save_rmse_mape(dataname, res)

    print("Long term:")
    res, testForecast_mean, testOriginal, size, forecast_d_MC_argmax, testForecast_uq, testForecast_lq = forecast(
        model, testX[:, 0:1, :], testY[:, 0:1, :], test_len, MC_S=200)

    save_rmse_mape(dataname, res, "Long-term")

# %%
if dataname == 'Pacific':

    for station in [0, 840]:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4),
                                       sharex=True, gridspec_kw={'height_ratios': [3, 0.5]})

        ax1.plot(testOriginal[:, station], label='original')
        ax1.plot(testForecast_mean[:, station], label='predict')
        ax1.plot(testForecast_uq[:, station], color='grey', alpha=0.8)
        ax1.plot(testForecast_lq[:, station], color='grey', alpha=0.8)
        ax1.fill_between(np.arange(
            size), testForecast_uq[:, station], testForecast_lq[:, station], color='grey', alpha=0.4)
        ax1.legend()

        sns.heatmap(forecast_d_MC_argmax.reshape(1, -1), linewidth=0,
                    cbar=False, alpha=1, cmap=cmap, vmin=0, vmax=1, ax=ax2)

        plt.suptitle('%s #%i' % (dataname, station))
        plt.savefig(figdirectory+'Station %i' % (station)+'.png', format='png')
        plt.show()

    save_rmse_mape(dataname, res)

    print("Long term:")
    res, testForecast_mean, testOriginal, size, forecast_d_MC_argmax, testForecast_uq, testForecast_lq = forecast(
        model, testX[:, 0:1, :], testY[:, 0:1, :], test_len, MC_S=200)

    save_rmse_mape(dataname, res, "Long-term")

# %%
if dataname == 'Electricity':
    for station in [0, 24]:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4),
                                       sharex=True, gridspec_kw={'height_ratios': [3, 0.5]})
        ax1.plot(testOriginal[:, station], label='original')
        ax1.plot(testForecast_mean[:, station], label='predict')
        ax1.plot(testForecast_uq[:, station], color='grey', alpha=0.8)
        ax1.plot(testForecast_lq[:, station], color='grey', alpha=0.8)
        ax1.fill_between(np.arange(
            size), testForecast_uq[:, station], testForecast_lq[:, station], color='grey', alpha=0.4)
        ax1.legend()
        sns.heatmap(forecast_d_MC_argmax.reshape(1, -1), linewidth=0,
                    cbar=False, alpha=1, cmap=cmap, vmin=0, vmax=1, ax=ax2)

        plt.suptitle('%s #%i' % (dataname, station))
        plt.savefig(figdirectory+'Station %i' % (station)+'.png', format='png')
        plt.show()

    save_rmse_mape(dataname, res)
