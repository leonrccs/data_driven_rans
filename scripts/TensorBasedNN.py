from abc import ABC

import torch as th
import torch.nn as nn
import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch
import random

dtype = th.double


class TBNN(nn.Module, ABC):
    def __init__(self, d_in, h, d_out):
        super(TBNN, self).__init__()
        self.input_layer = nn.Linear(d_in, h)
        self.hidden_layer_1 = nn.Linear(h, h)
        self.hidden_layer_2 = nn.Linear(h, h)
        self.hidden_layer_3 = nn.Linear(h, h)
        self.hidden_layer_4 = nn.Linear(h, int(h / 2))
        self.hidden_layer_5 = nn.Linear(int(h / 2), int(h / 4))
        self.hidden_layer_6 = nn.Linear(int(h / 4), int(h / 8))
        self.output_linear = nn.Linear(int(h / 8), d_out)
        self.NN = nn.Sequential(self.input_layer,
                                nn.LeakyReLU(),
                                self.hidden_layer_1,
                                nn.LeakyReLU(),
                                self.hidden_layer_2,
                                nn.LeakyReLU(),
                                self.hidden_layer_3,
                                nn.LeakyReLU(),
                                self.hidden_layer_4,
                                nn.LeakyReLU(),
                                self.hidden_layer_5,
                                nn.LeakyReLU(),
                                self.hidden_layer_6,
                                nn.LeakyReLU(),
                                self.output_linear)

    def forward(self, inv, t):
        """
        forward pass of the model. NN gives coefficients g which form b as a linear combination with t
        n is number of points fed to the NN
        :param inv: (dtype) invariants [n x 5]
        :param t: (dtype) tensor basis [n x 10 x 9]
        :return: b (dtype) anisotropy tensor [n x 9]
        :return: g (dtype) coefficients [n x 10]
        """
        g = self.NN(inv)
        g0 = g.unsqueeze(2).expand(inv.shape[0], 10, 9)
        return (g0 * t).sum(dim=1), g

    def reset_parameters(self):
        """
        Resets the weights of the neural network, samples from a normal guassian.
        """
        for x in self.modules():
            if isinstance(x, th.nn.Linear):
                x.weight.data = th.normal(th.zeros(x.weight.size()), th.zeros(x.weight.size()) + 0.2).type(dtype)
                x.bias.data = th.zeros(x.bias.size()).type(dtype)

    def save_nn(self, path):
        th.save(self.NN.state_dict(), path)

    def save_scripted_nn(self, path):
        """
        create traced instance of sequential nn model
        :param path: (str) path to write scripted model
        """
        traced_script_module = th.jit.trace(self.NN, th.rand(1, 5).type(dtype))
        traced_script_module.save(path)


class TBNN_generic(nn.Module, ABC):
    """
    An implementation of a fully connected feed forward
    Neural network in pytorch.
    """

    def __init__(self, layersizes=None,
                 activation=None,
                 final_layer_activation=None):
        """
        INPUTS:
            layersizes <list/tuple>: An iterable ordered object containing
                                 the sizes of the tensors from the
                                 input layer to the final output.
                                 (See example below).
            activation <callable>: A python callable through which
                                    torch backpropagation is possible.
            final_layer_activation <callable>: A python callable for
                                    the final layer activation function.
                                    Default: None (for regression problems)

        EXAMPLE:
            To define a NN with an input of size 2, 2 hidden layers of size
            50 and 50, output of size 1, with tanh activation function:
            >> layers = [2, 50, 50, 1]
            >> neuralnet = NeuralNet(layers, activation=torch.tanh)
            >> x = torch.randn(100, 2)   # 100 randomly sampled inputs
            >> output = neuralnet(x)  # compute the prediction at x.

        Inheriting from nn.Module ensures that all
        NN layers defined within the __init__ function are captured and
        stored in an OrderedDict object for easy accessibility.
        """
        super(TBNN_generic, self).__init__()
        if layersizes is None:
            layersizes = [1, 1]
        if activation is None:
            activation = th.relu
        self.layersizes = layersizes
        self.input_dim = self.layersizes[0]
        self.hidden_sizes = self.layersizes[1:-1]
        self.output_dim = self.layersizes[-1]
        self.activation = activation
        self.final_layer_activation = final_layer_activation
        if self.final_layer_activation is None:
            self.final_layer_activation = nn.Identity()
        self.nlayers = len(self.hidden_sizes) + 1
        self.layernames = []  # List to store all the FC layers

        # define FC layers
        for i in range(self.nlayers):
            layername = 'fc_{}'.format(i + 1)
            layermodule = nn.Linear(self.layersizes[i], self.layersizes[i + 1])
            self.layernames.append(layername)
            setattr(self, layername, layermodule)

    def forward(self, inv, t):
        """
        Implement the forward pass of the NN.
        """
        for i, layername in enumerate(self.layernames):
            fclayer = getattr(self, layername)
            inv = fclayer(inv)
            if i == self.nlayers - 1:
                inv = self.final_layer_activation(inv)
            else:
                inv = self.activation(inv)

        g = inv.unsqueeze(2).expand(inv.shape[0], 10, 9)
        return (g * t).sum(dim=1), g


def loss_realizability(tensor):

    # reshape tensor
    tensor = tensor.reshape(-1, 3, 3)

    diag_min = th.min(tensor[:, [0, 1, 2], [0, 1, 2]], 1)[0].unsqueeze(1)
    labels = (diag_min < th.tensor(-1. / 3.))

    # b_ii > -1/3
    loss_bii = nn.ReLU()(th.tensor(-1. / 3.) - diag_min)

    # 2*|b_ij| < b_ii + b_jj + 2/3
    loss_b12 = nn.ReLU()(2 * th.abs(tensor[:, 0, 1]) - (tensor[:, 0, 0] + tensor[:, 1, 1] + 2. / 3.))
    loss_b23 = nn.ReLU()(2 * th.abs(tensor[:, 1, 2]) - (tensor[:, 1, 1] + tensor[:, 2, 2] + 2. / 3.))
    loss_b13 = nn.ReLU()(2 * th.abs(tensor[:, 0, 2]) - (tensor[:, 0, 0] + tensor[:, 2, 2] + 2. / 3.))

    # lambda_1 > (3*|lambda_2| - lambda_2)/2,    lambda_1 < 1/3 - lambda_2
    eigval, eigvec = th.symeig(tensor, eigenvectors=True)
    loss_e1 = nn.ReLU()((3 * th.abs(eigval[:, 1]) - eigval[:, 1]) * .5 - eigval[:, 2])
    loss_e2 = nn.ReLU()(eigval[:, 2] - (1. / 3. - eigval[:, 1]))

    return th.sum(loss_bii + loss_b12 + loss_b23 + loss_b13 + loss_e1 + loss_e2)/tensor.shape[0]


class TBNNModel:
    def __init__(self, layersizes, activation, final_layer_activation):  # d_in, h, d_out):
        # self.model = TBNN(d_in, h, d_out).double()
        self.net = TBNN_generic(layersizes=layersizes,
                                activation=activation,
                                final_layer_activation=final_layer_activation).double()
        self.loss_fn = nn.MSELoss()
        self.loss_vector = np.array([])
        self.val_loss_vector = np.array([])
        self.inv = th.tensor([])
        self.inv_train = th.tensor([])
        self.inv_val = th.tensor([])
        self.mu = None
        self.std = None
        self.T = th.tensor([])
        self.T_train = th.tensor([])
        self.T_val = th.tensor([])
        self.b = th.tensor([])
        self.b_train = th.tensor([])
        self.b_val = th.tensor([])
        self.grid = th.tensor([])
        self.grid_train = th.tensor([])
        self.grid_val = th.tensor([])
        self.n_total = 0
        self.n_train = 0
        self.n_val = 0
        self.batch_size = 20

    def load_data(self, parent_dir, child_dir, n_samples=10000):
        """
        Method to load data in the model  (training and validation)
        :param n_samples: (int) number of samples per flow geometry
        :param parent_dir: (str) parent directory
        :param child_dir: (str) list of strings which hold directories to loop through
        """

        # loop over flow geometries
        for i, directory in enumerate(parent_dir):
            inv = th.tensor([])
            t = th.tensor([])
            b = th.tensor([])
            grid = th.tensor([])

            # loop over reynolds numbers
            for _, case in enumerate(child_dir[i]):
                curr_dir = os.sep.join([directory, case])
                inv = th.cat((inv, th.load(os.sep.join([curr_dir, 'inv-torch.th']))))
                t = th.cat((t, th.load(os.sep.join([curr_dir, 't-torch.th'])).flatten(2)))
                b = th.cat((b, th.load(os.sep.join([curr_dir, 'b_dns-torch.th'])).flatten(1)))
                grid = th.cat((grid, th.load(os.sep.join([curr_dir, 'grid-torch.th']))))

            # check if number of samples per geom exceeds n_samples
            print('n_samples in {}: {}'.format(directory, inv.shape[0]))
            if inv.shape[0] > n_samples:
                print('too many samples, randomly select {} samples ...'.format(n_samples))
                perm = th.randperm(inv.shape[0])
                self.inv = th.cat((self.inv, inv[perm[:n_samples]]))
                self.T = th.cat((self.T, t[perm[:n_samples]]))
                self.b = th.cat((self.b, b[perm[:n_samples]]))
                self.grid = th.cat((self.grid, grid[perm[:n_samples]]))
            else:
                self.inv = th.cat((self.inv, inv))
                self.T = th.cat((self.T, t))
                self.b = th.cat((self.b, b))
                self.grid = th.cat((self.grid, grid))

        self.n_total = self.inv.shape[0]
        print('Successfully loaded {} data points'.format(self.n_total))

    def normalize_features(self, cap=2.0):
        """
        normalize training features to mu = 0, sigma = 1
        saves normalized data and mu, sigma for scaling of test data
        :param cap: cap invatiants at certain level
        """
        # calculate mean and standard deviation
        mu = th.mean(self.inv, 0)
        std = th.std(self.inv, 0)
        std = std + (std < 0.005)*1.0

        # normalize tensor
        self.inv = (self.inv - mu) / std

        # remove outliers
        self.inv[self.inv > cap] = cap
        self.inv[self.inv < -cap] = -cap

        # rescale tensors and recalculate mu and std from capped tensor
        self.inv = self.inv * std + mu

        # calculate
        self.mu = th.mean(self.inv, 0)
        self.std = th.std(self.inv, 0)
        self.std = self.std + (self.std < 0.005)

        # renormalize tensor after capping
        self.inv = (self.inv - self.mu) / self.std

    def select_training_data(self, train_ratio=0.7, seed=None):
        """
        split data into training and test data. set random_state to ensure getting same dataset for debugging
        :param seed: specify seed. otherwise will be chosen randomly
        :param train_ratio: (float) ratio of n_train/n_total = [0,1]
        """

        self.n_train = int(self.n_total * train_ratio) - int(self.n_total * train_ratio) % self.batch_size
        self.n_val = self.n_total - self.n_train
        (self.inv_train, self.inv_val,
         self.T_train, self.T_val,
         self.b_train, self.b_val,
         self.grid_train, self.grid_val) = train_test_split(self.inv, self.T, self.b, self.grid,
                                                            test_size=self.n_val,
                                                            random_state=seed) # TODO remove seed for actual training
        self.inv_train.requires_grad = True

    def l2loss(self):
        reg = th.tensor(0.)
        for m in self.net.modules():
            if hasattr(m, 'weight'):
                reg += m.weight.norm()
        return reg

    def loss_function(self, method):

        def mse_b_full(a, b):
            return nn.MSELoss()(a, b)

        def mse_b_unique(a, b):
            return nn.MSELoss()(a[:, [0, 1, 2, 4, 5, 8]], b[:, [0, 1, 2, 4, 5, 8]])

        if method == 'b_full':
            return mse_b_full
        if method == 'b_unique':
            return mse_b_unique

    def train_model(self, **kwargs):
        """
        train the model. inputs learning_rate and epochs must be given. batch_size is optional and is 20 by default
        training data set must be adjusted to new batch_size
        :param kwargs: set parameters for training process.
        Valid keywords are:
            lr_initial,
            n_epochs,
            batch_size,
            weight_decay,
            moving_average,
            lambda_real,
            lr_scheduler,
            lr_scheduler_dict
        """

        # print the training parameters
        for key in kwargs:
            print('{:<12} {:<20}'.format('Parameter:', key))
            print('{:<12}     {}'.format('    Value:', kwargs[key]))

        # read the training parameters from parameter dict
        if 'lr_initial' in kwargs:
            lr_initial = kwargs['lr_initial']
        else:
            lr_initial = 2.5e-5

        if 'n_epochs' in kwargs:
            n_epochs = kwargs['n_epochs']
        else:
            n_epochs = 500

        if 'batch_size' in kwargs:
            batch_size = kwargs['batch_size']
        else:
            batch_size = 100

        if 'weight_decay' in kwargs:
            weight_decay = kwargs['weight_decay']
        else:
            weight_decay = 0.0

        if 'moving_average' in kwargs:
            moving_average = kwargs['moving_average']
        else:
            moving_average = 5
        if 'lambda_real' in kwargs:
            lambda_real = kwargs['lambda_real']
        else:
            lambda_real = 0.0

        if 'error_method' in kwargs:
            error_method = kwargs['error_method']
        else:
            error_method = 'b_full'

        # set optimizer
        optimizer = th.optim.Adam(self.net.parameters(), lr=lr_initial)

        # set learning rate scheduler if defined in param dict
        if 'lr_scheduler' in kwargs:
            scheduler = kwargs['lr_scheduler'](optimizer, **kwargs['lr_scheduler_dict'])
            print('Using learning rate scheduler with parameters:')
            print(scheduler.state_dict())
        else:
            scheduler = None

        print('Start training model ...')
        print('Using {} data points for training'.format(self.inv_train.shape[0]))

        # initialize training loss
        perm = np.random.permutation(self.inv_train.shape[0])
        initial_inv = self.inv_train[perm[0:batch_size]]
        initial_t = self.T_train[perm[0:batch_size]]
        initial_b = self.b_train[perm[0:batch_size]]
        initial_pred, _ = self.net(initial_inv, initial_t)
        self.loss_vector = self.loss_fn(initial_pred, initial_b).detach().numpy()
        last_val_loss_avg = 100.

        # initialize validation loss
        b_val_pred, _ = self.net(self.inv_val, self.T_val)
        self.val_loss_vector = self.loss_fn(b_val_pred, self.b_val).detach().numpy()

        # run loop over all epochs
        for epoch in range(n_epochs):

            # random permutation of training data before every epoch
            #     N_train = inv_train.shape[0]
            perm = np.random.permutation(self.n_train)

            # loop over all batches in one epoch
            for it in range(0, self.n_train, batch_size):
                # get batch data
                idx = perm[np.arange(it, it + batch_size)]
                inv_batch = self.inv_train[idx, :]
                t_batch = self.T_train[idx, :, :]
                b_batch = self.b_train[idx, :]

                # Forward pass
                b_pred, _ = self.net(inv_batch, t_batch)

                # compute violation constraint
                loss_real_train = loss_realizability(b_pred)

                # compute and print loss
                # loss = self.loss_fn(b_pred, b_batch) + weight_decay*self.l2loss() + lambda_real*loss_real_train
                loss = self.loss_function(error_method)(b_pred, b_batch)

                # print('b_unique: {}'.format(self.loss_function(error_method)(b_pred, b_batch)))
                # print('b_full:   {}'.format(self.loss_function('b_full')(b_pred, b_batch)))

                # reset gradient buffer
                optimizer.zero_grad()

                # get gradient
                loss.backward()

                # optimization step
                optimizer.step()

            # append loss to loss vector
            self.loss_vector = np.append(self.loss_vector, loss.detach().numpy())

            # compute validation error
            b_val_pred, _ = self.net(self.inv_val, self.T_val)
            # loss_val = (self.loss_fn(b_val_pred, self.b_val)
            #             + self.l2loss(weight_decay)
            #             + loss_realizability(b_val_pred))
            self.val_loss_vector = np.append(self.val_loss_vector,
                                             self.loss_fn(b_val_pred,self.b_val).detach().numpy())
            # self.val_loss_vector = np.append(self.val_loss_vector, loss_val.detach().numpy())
            # output optimization state
            if epoch % 20 == 0:
                print('Epoch: {}\n'.format(epoch),
                      'Training loss:              {:.6e}\n'.format(loss.item()),
                      'Validation loss:            {:.6e}\n'.format(self.val_loss_vector[-1]),
                      'l2 loss:                    {:.6e}\n'.format(weight_decay*self.l2loss()),
                      'Realizability loss (train): {:.6e}\n'.format(lambda_real*loss_realizability(b_pred)),
                      'Realizability loss (val):   {:.6e}\n'.format(loss_realizability(lambda_real*b_val_pred)))
                # check if learning rate should be updated
                if scheduler is None:
                    pass
                else:
                    print(scheduler.state_dict()['_last_lr'])

            # check if moving average decreased
            if (epoch > 500) & (epoch % 10 == 0) & kwargs['early_stopping']:
                this_val_loss_avg = np.sum(self.val_loss_vector[-moving_average:])/moving_average
                if this_val_loss_avg > last_val_loss_avg:
                    print('Validation loss moving average increased!')
                    print('Preliminary stop the optimization at epoch {}'.format(epoch))
                    print('Last validation loss average: {}'.format(last_val_loss_avg))
                    print('This validation loss average: {}'.format(this_val_loss_avg))
                    break
                else:
                    last_val_loss_avg = this_val_loss_avg

            # scheduler step
            if scheduler is None:
                pass
            elif 'step_arg' in kwargs:
                scheduler.step(self.val_loss_vector[-1])
            else:
                scheduler.step()


if __name__ == '__main__':

    training_dir = '/home/leonriccius/Documents/Fluid_Data/training_data'
    training_flow_geom = ['periodic_hills', 'conv_div_channel']
    training_cases = [['700', '1400', '5600', '10595'], ['7900', '12600']]
    tensor_basis = 'tensordata_norm_corr_t10'
    training_dirs = [os.sep.join([training_dir, geom, tensor_basis]) for geom in training_flow_geom]
    ling_architecture = [5, 30, 30, 30, 30, 30, 30, 30, 30, 10]
    layers = ling_architecture
    model = TBNNModel(layersizes=layers, activation=nn.LeakyReLU(), final_layer_activation=th.tanh)

    model.load_data(training_dirs, training_cases, n_samples=10000)