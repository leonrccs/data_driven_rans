from scripts import TensorBasedNN
import torch as th
import torch.nn as nn
import os
import random

if __name__ == '__main__':

    training_dir = '/home/leonriccius/Documents/Fluid_Data/tensordata'
    training_flow_geom = ['PeriodicHills', 'ConvDivChannel']
    training_cases = [['700', '1400', '5600', '10595'], ['12600']]
    training_dirs = [os.sep.join([training_dir, geom]) for geom in training_flow_geom]


    geneva_architecture = [5, 200, 200, 200, 40, 20, 10]
    ling_architecture = [5, 30, 30, 30, 30, 30, 30, 30, 30, 10]
    layers = ling_architecture
    model = TensorBasedNN.TBNNModel(layersizes=layers,
                                    activation=nn.LeakyReLU(),
                                    final_layer_activation=th.tanh)

    save_model_path = './storage/models/kaandorp_data/ph_cdc/20000dp_1000ep_100bs_2-5e-5lr'
    assert os.path.exists(save_model_path), 'path to save model not specified'

    # model.net = th.load(model_path)

    for m in model.net.modules():
        print(m)

    model.load_data(training_dirs, training_cases)
    batch_size = 100
    model.batch_size = batch_size  # need to specify it here because select train data needs it

    seed = random.seed()
    model.select_training_data(train_ratio=0.7)

    # lr_scheduler_opt = {'mode': 'min', 'factor': 0.75, 'patience': 3, 'verbose': True,
    #                     'threshold': 0.05, 'threshold_mode': 'rel',
    #                     'cooldown': 5, 'min_lr': '2.5e-6', 'eps': 1e-07}
    lr_scheduler_ = th.optim.lr_scheduler.MultiStepLR
    lr_scheduler_opt = {'milestones': [25, 50, 100, 200, 400, 700], 'gamma': .5, 'verbose': False}
    # model.train_model(lr_initial=0.00016, n_epochs=1000, batch_size=50,
    #                   lr_scheduler=lr_scheduler_, **lr_scheduler_opt)

    model.train_model(lr_initial=2.5e-5, n_epochs=1000, batch_size=batch_size)

    th.save(model.net, os.sep.join([save_model_path, 'model.pt']))
    th.save(model.net.state_dict(), os.sep.join([save_model_path, 'state_dict.pt']))
    th.save(model.loss_vector, os.sep.join([save_model_path, 'loss_vector.th']))
    th.save(model.val_loss_vector, os.sep.join([save_model_path, 'val_loss_vector.th']))
    # model.net.reset_parameters()


    #how to load data before:
    # training_dir = '/home/leonriccius/Documents/Fluid_Data/training_data'
    # # workstation_mount_point = '/home/leonriccius/gkm'  # put your mouthpoint of workstation here
    # # training_dir = os.sep.join([workstation_mount_point, 'Masters_Thesis/Fluid_Data/training_data']) #
    # training_flow_geom = ['periodic_hills', 'conv_div_channel']
    # training_cases = [['700', '1400', '5600', '10595'], ['7900', '12600']]  # , '10595']
    # # training_dir = '/home/leonriccius/Documents/Fluid_Data/training_data/periodic_hills/tensordata'
    # # training_cases = ['700', '1400', '5600'] #, '10595']
    #
    # training_dirs = [os.sep.join([training_dir, geom, 'tensordata_norm_corr_t10']) for geom in training_flow_geom]
    # # select what type of data you want. see training data folder for options
    # print(training_dirs)

















    # NN = TensorBasedNN.TBNNModel(d_in, H, d_out)
    #
    # NN.model.load_state_dict(th.load(model_path))
    # NN.model.eval()
    #
    # NN.load_data(training_dir, training_cases)
    # NN.select_training_data(train_ratio=0.7)
    #
    # new_model_path = './storage/models/test/nn_traced.pt'
    # # NN.model.save_nn(new_model_path)

    # NN.train_model(learning_rate=1e-6, n_epochs=500, batch_size=20)

    # new_model_path = './storage/models/test'
    # th.save(NN.model, os.sep.join([new_model_path, 'model.pt']))
    # th.save(NN.model.state_dict(), os.sep.join([new_model_path, 'state_dict.pt']))
    # th.save(NN.loss_vector, os.sep.join([new_model_path, 'loss_vector.th']))
    # th.save(NN.val_loss_vector, os.sep.join([new_model_path, 'val_loss_vector.th']))
    # NN.model.reset_parameters()
