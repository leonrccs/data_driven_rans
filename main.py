from scripts import TensorBasedNN
import torch as th
import torch.nn as nn
import os
import random

if __name__ == '__main__':

    training_dir = '/home/leonriccius/Documents/Fluid_Data/tensordata_fs1_fs2_fs3_reduced'
    training_flow_geom = ['ConvDivChannel']  # ['PeriodicHills', 'ConvDivChannel']
    training_cases =  [['7900']] # [['5600', '10595'], ['12600']] # [['2000', '2400', '2900', '3200']]
    # training_flow_geom = ['ConvDivChannel']
    # training_cases = [['7900']]
    # training_flow_geom = ['PeriodicHills']
    # training_cases = [['5600']]
    training_dirs = [os.sep.join([training_dir, geom]) for geom in training_flow_geom]

    weight_decay = 10. ** th.linspace(-14, 0, 8)
    # weight_decay = [0.0]
    print(weight_decay)

    for weight_decay_ in weight_decay:

        # set training features and network architecture
        features = ['FS1', 'FS2', 'FS3']
        n_features = 17  # fs_1: 3,  fs_2: 5,  fs:3: 9
        geneva_architecture = [n_features, 200, 200, 200, 40, 20, 10]
        ling_architecture = [n_features, 30, 30, 30, 30, 30, 30, 30, 30, 10]
        layers = ling_architecture

        # set seed and initialize model
        th.manual_seed(12345)
        model = TensorBasedNN.TBNNModel(layersizes=layers,
                                        activation=nn.LeakyReLU(),
                                        final_layer_activation=nn.Tanh())

        print(model.net.fc_1.weight[0])
        # nn.LeakyReLU() usually used for activation, th.tanh usually used for final layer

        # # if pretrained model should be used
        # model.net = th.load(model_path)
        print('______________________________________________________________________________________________')
        print('NN initialized')

        # save_model_path = './storage/models/kaandorp_data/ph_cdc/l2_regularization_1000ep_1000_bs/{:.0e}'.format(weight_decay_)
        # save_model_path = './storage/models/kaandorp_data/ph_cdc_sd/invariants_corrected/tanh_activation/1000_epochs'
        # base_path = './storage/models/kaandorp_data/ph_cdc_sd/additional_features/sd_2000_2400_2900_3200'
        base_path = './storage/models/kaandorp_data/weight_decay/cdc7900/builtin'
        save_model_path = os.sep.join([base_path, '{:.0e}'.format(weight_decay_)])
        # save_model_path = base_path
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)

        assert os.path.exists(save_model_path), 'path to save model not specified'
        save_model = True

        # # print net modules
        # for m in model.net.modules():
        #     print(m)

        model.load_data(training_dirs, training_cases, features, n_samples=14000)
        model.normalize_features(cap=2.)
        batch_size = 100
        model.batch_size = batch_size  # need to specify it here because select train data needs it

        # select data for training
        model.select_training_data(train_ratio=0.7, seed=12345)

        # # in case learning rate scheduler is used
        # lr_scheduler_opt = {'mode': 'min', 'factor': 0.75, 'patience': 3, 'verbose': True,
        #                     'threshold': 0.05, 'threshold_mode': 'rel',
        #                     'cooldown': 5, 'min_lr': '2.5e-6', 'eps': 1e-07}
        # lr_scheduler = th.optim.lr_scheduler.MultiStepLR
        # lr_scheduler_opt = {'milestones': [25, 50, 100, 200, 400, 700], 'gamma': .5, 'verbose': False}
        # model.train_model(lr_initial=0.00016, n_epochs=1000, batch_size=50,
        #                   lr_scheduler=lr_scheduler_, **lr_scheduler_opt)

        learning_rate = 2.5e-5
        # parameters for model training
        training_params = {'lr_initial': learning_rate, 'n_epochs': 1000, 'batch_size': batch_size,
                           'fixed_seed': True,
                           'early_stopping': True, 'moving_average': 5, 'min_epochs': 200,
                           'weight_decay': weight_decay_, 'builtin_weightdecay': True,
                           'lambda_real': 0.0,
                           'error_method': 'b_unique'}
                           # 'lr_scheduler': lr_scheduler, 'lr_scheduler_dict': lr_scheduler_opt}

        model.train_model(**training_params)
        # model.train_model(lr_initial=2.5e-5,
        #                   n_epochs=1000,
        #                   batch_size=batch_size)  #,  weight_decay=weight_decay_)

        print(training_params)
        print('Saving model and relevant data ...')

        if save_model:
            th.save(model.net, os.sep.join([save_model_path, 'model.pt']))
            th.save(model.net.state_dict(), os.sep.join([save_model_path, 'state_dict.pt']))
            th.save(model.mu, os.sep.join([save_model_path, 'mu.th']))
            th.save(model.std, os.sep.join([save_model_path, 'std.th']))
            th.save(training_params, os.sep.join([save_model_path, 'training_params.th']))
            th.save(model.loss_vector, os.sep.join([save_model_path, 'loss_vector.th']))
            th.save(model.val_loss_vector, os.sep.join([save_model_path, 'val_loss_vector.th']))
            # model.net.reset_parameters()

        print('... Done!\n')

        del model
