from abc import ABC

import torch as th
import torch.nn as nn
import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch

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

class TBNNModel:
    def __init__(self, d_in, h, d_out):
        self.model = TBNN(d_in, h, d_out).double()
        self.loss_fn = nn.MSELoss()
        self.loss_vector = np.array([])
        self.val_loss_vector = np.array([])
        self.inv = th.tensor([])
        self.inv_train = th.tensor([])
        self.inv_val = th.tensor([])
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

    def load_data(self, parent_dir, child_dir):
        """
        Method to load data in the model  (training and validation)
        :param parent_dir: (str) parent directory
        :param child_dir: (str) list of strings which hold directories to loop through
        """
        for i, case in enumerate(child_dir):
            curr_dir = os.sep.join([parent_dir, case])
            self.inv = th.cat((self.inv, th.load(os.sep.join([curr_dir, 'inv-torch.th']))))
            self.T = th.cat((self.T, th.load(os.sep.join([curr_dir, 'T-torch.th'])).flatten(2)))
            self.b = th.cat((self.b, th.load(os.sep.join([curr_dir, 'b-torch.th'])).flatten(1)))
            self.grid = th.cat((self.grid, th.load(os.sep.join([curr_dir, 'grid-torch.th']))))

        self.n_total = self.inv.shape[0]

    def select_training_data(self, train_ratio=0.7):
        """
        split data into training and test data. set random_state to ensure getting same dataset for debugging
        :param train_ratio: (float) ratio of n_train/n_total = [0,1]
        """
        self.n_train = int(self.n_total * train_ratio) - int(self.n_total * train_ratio) % self.batch_size
        self.n_val = self.n_total - self.n_train
        (self.inv_train, self.inv_val,
         self.T_train, self.T_val,
         self.b_train, self.b_val,
         self.grid_train, self.grid_val) = train_test_split(self.inv, self.T, self.b, self.grid,
                                                            test_size=self.n_val,
                                                            random_state=42) # TODO remove seed for actual training
        self.inv_train.requires_grad = True

    def l2loss(self, lmbda):
        reg = th.tensor(0.)
        for m in self.model.modules():
            if hasattr(m, 'weight'):
                reg += m.weight.norm() ** 20
        return lmbda * reg

    def train_model(self, learning_rate, n_epochs=500, batch_size=20):
        """
        train the model. inputs learning_rate and epochs must be given. batch_size is optional and is 20 by default
        training data set must be adjusted to new batch_size
        :param learning_rate: (float) learning rate of the optimization
        :param n_epochs: (int) number of epochs for training
        :param batch_size: (int) batch size for each optimization step
        """
        self.select_training_data()

        # set optimizer
        optimizer = th.optim.Adam(self.model.parameters(), lr=learning_rate)

        # initialize training loss
        perm = np.random.permutation(self.inv_train.shape[0])
        initial_inv = self.inv_train[perm[0:20]]
        initial_T = self.T_train[perm[0:20]]
        initial_b = self.b_train[perm[0:20]]
        initial_pred, _ = self.model(initial_inv, initial_T)
        self.loss_vector = self.loss_fn(initial_pred, initial_b).detach().numpy()

        # initialize validation loss
        b_val_pred, _ = self.model(self.inv_val, self.T_val)
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
                T_batch = self.T_train[idx, :, :]
                b_batch = self.b_train[idx, :]

                # Forward pass
                b_pred, _ = self.model(inv_batch, T_batch)

                # compute and print loss
                loss = self.loss_fn(b_pred, b_batch)  # + L2loss(lmbda, model)

                # reset gradient buffer
                optimizer.zero_grad()

                # get gradient
                loss.backward()

                # optimization step
                optimizer.step()

            # append loss to loss vector
            self.loss_vector = np.append(self.loss_vector, loss.detach().numpy())

            # compute validation error
            b_val_pred, _ = self.model(self.inv_val, self.T_val)
            self.val_loss_vector = np.append(self.val_loss_vector, self.loss_fn(b_val_pred, self.b_val).detach().numpy())

            # output optimization state
            if epoch % 20 == 0:
                print('Epoch: {}, Training loss: {:.6f}, Validation loss {:.6f}'.format(epoch, loss.item(),
                                                                                        self.val_loss_vector[-1]))
