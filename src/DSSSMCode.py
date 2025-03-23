import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class DSSSM(nn.Module):

    def __init__(self, x_dim, y_dim, h_dim, z_dim, d_dim, n_layers, device, bidirection=False, bias=False, dataname=None):

        super(DSSSM, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.d_dim = d_dim
        self.n_layers = n_layers
        self.device = device
        self.temperature = 0.5
        self.bidirection = bidirection
        self.dataname = dataname

        self.Transition_initial = torch.eye(self.d_dim, device=self.device) * (
            1-0.05*self.d_dim) + torch.ones((self.d_dim, self.d_dim), device=self.device) * 0.05

        # d prior (trainsition matrix)
        self.dprior = nn.Sequential(
            nn.Linear(d_dim, d_dim),
            nn.Softmax(dim=1))

        # z trainsition (z prior)
        self.ztrainsition_list = nn.ModuleList()
        self.ztrainsition_mean_list = nn.ModuleList()
        self.ztrainsition_std_list = nn.ModuleList()

        # d posterior
        self.dposterior_list = nn.ModuleList()

        # z posterior
        self.zposterior_list = nn.ModuleList()
        self.zposterior_mean_list = nn.ModuleList()
        self.zposterior_std_list = nn.ModuleList()

        self.yemission_list = nn.ModuleList()
        self.yemission_mean_list = nn.ModuleList()
        self.yemission_std_list = nn.ModuleList()

        for i in range(self.d_dim):

            dposterior = nn.Sequential(
                nn.Linear(h_dim, d_dim),
                nn.Softmax(dim=1))
            self.dposterior_list.append(dposterior)

            zposterior = nn.Sequential(
                nn.Linear(z_dim + h_dim, z_dim),
                nn.ReLU(),
                nn.Linear(z_dim, z_dim),
                nn.ReLU())
            zposterior_mean = nn.Linear(z_dim, z_dim)
            zposterior_std = nn.Sequential(
                nn.Linear(z_dim, z_dim),
                nn.Softplus())
            self.zposterior_list.append(zposterior)
            self.zposterior_mean_list.append(zposterior_mean)
            self.zposterior_std_list.append(zposterior_std)

            ztrainsition = nn.Sequential(
                nn.Linear(z_dim + h_dim, z_dim),
                nn.ReLU(),
                nn.Linear(z_dim, z_dim),
                nn.ReLU())
            ztrainsition_mean = nn.Linear(z_dim, z_dim)
            ztrainsition_std = nn.Sequential(
                nn.Linear(z_dim, z_dim),
                nn.Softplus())
            self.ztrainsition_list.append(ztrainsition)
            self.ztrainsition_mean_list.append(ztrainsition_mean)
            self.ztrainsition_std_list.append(ztrainsition_std)

            yemission = nn.Sequential(
                nn.Linear(z_dim + h_dim, y_dim),
                nn.ReLU(),
                nn.Linear(y_dim, y_dim),
                nn.ReLU())
            yemission_mean = nn.Linear(y_dim, y_dim)
            yemission_std = nn.Sequential(
                nn.Linear(y_dim, y_dim),
                nn.Softplus())
            self.yemission_list.append(yemission)
            self.yemission_mean_list.append(yemission_mean)
            self.yemission_std_list.append(yemission_std)

        # recurrence
        self.rnn_forward = nn.GRU(x_dim, h_dim, n_layers, bidirectional=False)

        if self.bidirection:
            self.rnn_backward = nn.GRU(
                y_dim + h_dim, int(h_dim/2), n_layers, bidirectional=self.bidirection)
        else:
            self.rnn_backward = nn.GRU(
                y_dim + h_dim, h_dim, n_layers, bidirectional=self.bidirection)

    def TransitionMatrix(self):

        if self.dataname == 'Sleep':
            Transition = self.dprior(
                self.Transition_initial)*0.2 + torch.eye(self.d_dim, device=self.device)*0.8
        else:
            Transition = self.dprior(
                self.Transition_initial)/2 + torch.eye(self.d_dim, device=self.device)/2

        return Transition

    def forward(self, x, y):

        Transition = self.TransitionMatrix()

        all_d_posterior = []  # probability vector
        all_d_t_sampled_plot = []  # 0,1
        all_d_t_sampled = []  # one-hot vector

        all_z_posterior_mean, all_z_posterior_std = [], []
        all_z_t_sampled = []

        all_y_emission_mean, all_y_emission_std = [], []

        kld_gaussian_loss = 0
        kld_category_loss = 0
        nll_loss = 0

        h0 = torch.zeros((self.n_layers, x.size(
            1), self.h_dim), device=self.device)
        if self.bidirection:
            A0 = torch.zeros((self.n_layers*2, x.size(1),
                             int(self.h_dim/2)), device=self.device)
        else:
            A0 = torch.zeros((self.n_layers, x.size(
                1), self.h_dim), device=self.device)

        samples = torch.distributions.Categorical(
            torch.ones((self.d_dim))/self.d_dim).sample((x.size(1),)).type(torch.LongTensor)
        d0 = self._one_hot_encode(samples, self.d_dim)
        all_d_posterior.append(torch.ones(
            (x.size(1), self.d_dim), device=self.device)/self.d_dim)
        all_d_t_sampled_plot.append(samples.reshape(-1, 1).to(self.device))
        all_d_t_sampled.append(d0)

        z0 = torch.zeros((x.size(1), self.z_dim), device=self.device)
        all_z_posterior_std.append(z0)
        all_z_posterior_mean.append(z0)
        all_z_t_sampled.append(z0)

        # Forward RNN
        output_forward, h_forward = self.rnn_forward(x, h0)

        # Backward Rnn
        # Concatenate y and h
        yh_concatenate = torch.cat([y, output_forward], 2)
        # Reverse of copy of numpy array of given tensor
        yh_concatenate_inverse = torch.flip(yh_concatenate, [0])
        output_backward, h_backward = self.rnn_backward(
            yh_concatenate_inverse, A0)

        for t in range(x.size(0)):

            # d prior
            d_prior = torch.mm(all_d_t_sampled[t], Transition)  # 1*d_dim

            # d posterior
            d_posterior_list = []
            d_posterior = 0
            for i in range(self.d_dim):
                d_posterior_list.append(self.dposterior_list[i](
                    output_backward[x.size(0)-t-1]))  # batch*2
                d_posterior += d_posterior_list[i] * \
                    all_d_t_sampled[t][:, i:(i+1)]
            # output_backward.size() is timestep*batch*h_dim
            all_d_posterior.append(d_posterior)

            d_t_samples = torch.distributions.Categorical(
                d_posterior).sample().type(torch.LongTensor).to(self.device)
            all_d_t_sampled_plot.append(d_t_samples.reshape(-1, 1))
            all_d_t_sampled.append(
                self._one_hot_encode(d_t_samples, self.d_dim))

            # z prior
            z_prior_list = []
            z_prior_mean_list = []
            z_prior_std_list = []
            z_prior_mean = 0
            z_prior_std = 0

            # z posterior
            z_posterior_list = []
            z_posterior_mean_list = []
            z_posterior_std_list = []
            z_posterior_mean = 0
            z_posterior_std = 0

            for i in range(self.d_dim):
                z_prior_list.append(self.ztrainsition_list[i](
                    torch.cat([output_forward[t], all_z_t_sampled[t]], 1)))
                z_prior_mean_list.append(
                    self.ztrainsition_mean_list[i](z_prior_list[i]))
                z_prior_std_list.append(
                    self.ztrainsition_std_list[i](z_prior_list[i]))
                z_prior_mean += z_prior_mean_list[i] * \
                    all_d_t_sampled[t+1][:, i:(i+1)]
                z_prior_std += z_prior_std_list[i] * \
                    all_d_t_sampled[t+1][:, i:(i+1)]

                z_posterior_list.append(self.zposterior_list[i](
                    torch.cat([output_backward[x.size(0)-t-1], all_z_t_sampled[t]], 1)))
                z_posterior_mean_list.append(
                    self.zposterior_mean_list[i](z_posterior_list[i]))
                z_posterior_std_list.append(
                    self.zposterior_std_list[i](z_posterior_list[i]))
                z_posterior_mean += z_posterior_mean_list[i] * \
                    all_d_t_sampled[t+1][:, i:(i+1)]
                z_posterior_std += z_posterior_std_list[i] * \
                    all_d_t_sampled[t+1][:, i:(i+1)]

            all_z_posterior_mean.append(z_posterior_mean)
            all_z_posterior_std.append(z_posterior_std)

            # sampling and reparameterization for the continuous variable
            z_t = self._reparameterized_normal_sample(
                z_posterior_mean, z_posterior_std)
            all_z_t_sampled.append(z_t)

            # y emission
            y_emission_list = []
            y_emission_mean_list = []
            y_emission_std_list = []
            y_emission_mean = 0
            y_emission_std = 0

            for i in range(self.d_dim):

                y_emission_list.append(self.yemission_list[i](
                    torch.cat([output_forward[t], all_z_t_sampled[t+1]], 1)))
                y_emission_mean_list.append(
                    self.yemission_mean_list[i](y_emission_list[i]))
                y_emission_std_list.append(
                    self.yemission_std_list[i](y_emission_list[i]))
                y_emission_mean += y_emission_mean_list[i] * \
                    all_d_t_sampled[t+1][:, i:(i+1)]
                y_emission_std += y_emission_std_list[i] * \
                    all_d_t_sampled[t+1][:, i:(i+1)]

            all_y_emission_mean.append(y_emission_mean)
            all_y_emission_std.append(y_emission_std)

            # computing losses
            for i in range(self.d_dim):
                kld_gaussian_loss += torch.sum(
                    self._kld_gauss(z_posterior_mean_list[i], z_posterior_std_list[i],
                                    z_prior_mean_list[i], z_prior_std_list[i]) * d_posterior[:, i:(i+1)])

            for i in range(self.d_dim):
                kld_category_loss += torch.sum(self._kld_category(
                    d_posterior_list[i], Transition[i:(i+1), :]) * all_d_posterior[-2][:, i])

            for i in range(self.d_dim):
                nll_loss += torch.sum(self._nll_gauss(
                    y_emission_mean_list[i], y_emission_std_list[i], y[t])*d_posterior[:, i:(i+1)])

        return kld_gaussian_loss, kld_category_loss, nll_loss, (all_z_posterior_mean, all_z_posterior_std), (all_y_emission_mean, all_y_emission_std), all_d_t_sampled_plot, all_z_t_sampled, all_d_posterior, all_d_t_sampled


    def _forecastingMultiStep(self, x, y, step=1, S=1):

        with torch.no_grad():

            Transition = self.TransitionMatrix()
            h0 = torch.zeros((self.n_layers, x.size(
                1), self.h_dim), device=self.device)
            output_forward, h_forward = self.rnn_forward(x, h0)

            forecast_MC = []
            forecast_d_MC = []
            forecast_z_MC = []

            for s in range(S):

                kld_gaussian_loss, kld_category_loss, nll_loss, (all_z_posterior_mean, all_z_posterior_std), (
                    all_y_emission_mean, all_y_emission_std), all_d_t_sampled_plot, z_t_sampled, all_d_posterior, all_d_t_sampled = self.forward(x, y)

                forecast_x = []
                forecast_y = []
                forecast_x.append(y[-1, :, :].unsqueeze(0))

                for t in range(step):

                    _, h_forward = self.rnn_forward(forecast_x[t], h_forward)

                    # d prior
                    d_prior = torch.mm(all_d_t_sampled[-1], Transition)

                    # sample d from the prior
                    samples = torch.distributions.Categorical(
                        d_prior).sample().type(torch.LongTensor)
                    d_t_sampled = self._one_hot_encode(samples, self.d_dim)
                    all_d_t_sampled_plot.append(
                        samples.reshape(-1, 1).to(self.device))
                    all_d_t_sampled.append(d_t_sampled)
                    all_d_posterior.append(d_prior)

                    # z prior
                    z_prior_list = []
                    z_prior_mean_list = []
                    z_prior_std_list = []
                    z_prior_mean = 0
                    z_prior_std = 0
                    for i in range(self.d_dim):
                        z_prior_list.append(self.ztrainsition_list[i](
                            torch.cat([h_forward.squeeze(0), z_t_sampled[-1]], 1)))
                        z_prior_mean_list.append(
                            self.ztrainsition_mean_list[i](z_prior_list[i]))
                        z_prior_std_list.append(
                            self.ztrainsition_std_list[i](z_prior_list[i]))
                        z_prior_mean += z_prior_mean_list[i] * \
                            d_t_sampled[:, i:(i+1)]
                        z_prior_std += z_prior_std_list[i] * \
                            d_t_sampled[:, i:(i+1)]

                    # sample z
                    z_t = torch.distributions.Normal(
                        z_prior_mean, z_prior_std).sample()
                    z_t_sampled.append(z_t)

                    all_z_posterior_mean.append(z_prior_mean)
                    all_z_posterior_std.append(z_prior_std)

                    # y emission
                    y_emission_list = []
                    y_emission_mean_list = []
                    y_emission_std_list = []
                    y_emission_mean = 0
                    y_emission_std = 0

                    for i in range(self.d_dim):
                        y_emission_list.append(self.yemission_list[i](
                            torch.cat([h_forward.squeeze(0), z_t_sampled[-1]], 1)))
                        y_emission_mean_list.append(self.yemission_mean_list[i](
                            y_emission_list[i]))
                        y_emission_std_list.append(self.yemission_std_list[i](
                            y_emission_list[i]))
                        y_emission_mean += y_emission_mean_list[i] * \
                            d_t_sampled[:, i:(i+1)]
                        y_emission_std += y_emission_std_list[i] * \
                            d_t_sampled[:, i:(i+1)]
                    all_y_emission_mean.append(y_emission_mean)
                    all_y_emission_std.append(y_emission_std)
                    y_t = torch.distributions.Normal(
                        y_emission_mean, y_emission_std).sample().unsqueeze(0)
                    forecast_x.append(y_t)
                    forecast_y.append(y_t.squeeze(0).cpu().numpy())

                forecast_MC.append(forecast_y)
                forecast_d_MC.append(all_d_t_sampled_plot)
                forecast_z_MC.append(z_t_sampled)

            forecast_MC = np.array(forecast_MC)
            forecast_z_MC = torch.stack(
                [torch.stack(forecast_z_MC[i]) for i in range(len(forecast_z_MC))]).cpu().numpy()
            forecast_d_MC = torch.stack(
                [torch.stack(forecast_d_MC[i]) for i in range(len(forecast_d_MC))]).cpu().numpy()

        return forecast_MC, forecast_d_MC, forecast_z_MC

    def _reparameterized_normal_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_()
        eps = eps.to(self.device)
        return eps.mul(std).add_(mean)

    def _reparameterized_category_gumbel_softmax_sample(self, logits):
        """using std to sample"""
        if self.temperature > 0.01:
            self.temperature = self.temperature/1.001
        else:
            self.temperature = 0.01
        y = torch.log(logits) + torch.distributions.Gumbel(torch.tensor(
            [0.0], device=self.device), torch.tensor([1.0], device=self.device)).sample(logits.size()).squeeze()
        return torch.nn.functional.softmax((y / self.temperature), dim=1)

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""
        # mean_1 posterior
        # mean_2 prior
        kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) +
                       (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
                       std_2.pow(2) - 1)
        return 0.5 * kld_element

    def _kld_category(self, d_posterior, d_prior):
        # Already sum up
        return torch.sum(torch.mul(torch.log(torch.div(d_posterior, d_prior)), d_posterior), axis=1)

    def _nll_bernoulli(self, theta, x):
        # Already sum up
        return - torch.sum(x*torch.log(theta) + (1-x)*torch.log(1-theta))

    def _nll_gauss(self, mean, std, x):
        return 0.5*torch.log(torch.tensor(2*math.pi, device=self.device))+torch.log(std) + (x-mean).pow(2)/(2*std.pow(2))

    def _one_hot_encode(self, x, n_classes):
        """One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
        x: List of sample Labels
        return: Numpy array of one-hot encoded labels
        """
        return torch.eye(n_classes, device=self.device)[x]


def train(model, optimizer, trainX, trainY, epoch, batch_size, n_epochs, status="train"):

    model.train()
    # forward + backward + optimize
    if epoch < n_epochs/2:
        annealing = 0.01
    else:
        annealing = min(1.0, 0.01 + epoch / n_epochs/2)
    # annealing = 0.01
    print('Annealing coef:', annealing)

    for batch in range(0, trainX.size(1), batch_size):

        batchX = trainX[:, batch:(batch+batch_size), :]
        batchY = trainY[:, batch:(batch+batch_size), :]
        kld_gaussian_loss, kld_category_loss, nll_loss, _, _, _, _, _, _ = model(
            batchX, batchY)
        kld_loss = (kld_gaussian_loss + kld_category_loss)
        loss = annealing * kld_loss / \
            (batchX.size(1)*batchX.size(0)) + \
            nll_loss/(batchX.size(1)*batchX.size(0))

        optimizer.zero_grad()
        loss.backward()
        # grad norm clipping, only in pytorch version >= 1.10
        if (batch == 0) & (epoch % 10 == 0):
            plot_grad_flow(model.named_parameters())
        optimizer.step()

    all_d_t_sampled_plot, all_z_t_sampled, loss, all_d_posterior, all_z_posterior_mean = test(
        model, trainX, trainY, epoch, "train")

    return all_d_t_sampled_plot, all_z_t_sampled, loss, all_d_posterior, all_z_posterior_mean


def test(model, testX, testY, epoch, status="test"):
    """uses test data to evaluate 
    likelihood of the model"""
    model.eval()
    with torch.no_grad():
        size = testX.size(1)*testX.size(0)

        kld_gaussian_loss, kld_category_loss, nll_loss, (all_z_posterior_mean, all_z_posterior_std), (
            all_y_emission_mean, all_y_emission_std), all_d_t_sampled_plot, all_z_t_sampled, all_d_posterior, all_d_t_sampled = model(testX, testY)
        nll_loss_total = nll_loss.item()
        kld_gaussian_loss_total = kld_gaussian_loss.item()
        kld_category_loss_total = kld_category_loss.item()

        loss = kld_gaussian_loss_total + kld_category_loss_total + nll_loss_total
        print('{} Epoch:{}\t KLD_Gaussian Loss: {:.6f}, KLD_Category Loss: {:.6f}, NLL Loss: {:.6f}, Loss: {:.4f}'.format(
            status, epoch, kld_gaussian_loss_total/size, kld_category_loss_total/size, nll_loss_total/size, loss/size))

        # size = (batch,timestep,dim)
        all_d_t_sampled = torch.stack(
            all_d_t_sampled).cpu().numpy().transpose((1, 0, 2))
        all_d_t_sampled_plot = torch.stack(
            all_d_t_sampled_plot).cpu().numpy().transpose((1, 0, 2))
        all_z_t_sampled = torch.stack(
            all_z_t_sampled).cpu().numpy().transpose((1, 0, 2))
        all_d_posterior = torch.stack(
            all_d_posterior).cpu().numpy().transpose((1, 0, 2))
        all_z_posterior_mean = torch.stack(
            all_z_posterior_mean).cpu().numpy().transpose((1, 0, 2))

    return all_d_t_sampled_plot, all_z_t_sampled, loss/size, all_d_posterior, all_z_posterior_mean


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):  # "bias" not in n
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), [g.cpu().numpy() for g in max_grads], alpha=0.2, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), [g.cpu().numpy() for g in ave_grads], alpha=0.2, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
