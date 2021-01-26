from scripts import TensorBasedNN
import torch as th
import torch.nn as nn
import os

if __name__ == '__main__':
    # training_dir = '/home/leonriccius/Documents/Fluid_Data/training_data'
    workstation_mount_point = '/home/leonriccius/gkm'  # put your mouthpoint of workstation here
    training_dir = os.sep.join([workstation_mount_point, 'Masters_Thesis/Fluid_Data/training_data']) #
    training_flow_geom = ['periodic_hills']  # ['periodic_hills', 'conv_div_channel']
    training_cases = [['700', '1400', '5600', '10595']]   #, ['7900', '12600']] #, '10595']
    # training_dir = '/home/leonriccius/Documents/Fluid_Data/training_data/periodic_hills/tensordata'
    # training_cases = ['700', '1400', '5600'] #, '10595']
    # model_path = './notebooks/trained_models/3cases/state_dict.pt'
    # model_path = './storage/models/periodic_hills/1000epochs/model.pt'

    # d_in, d_out = 5, 10
    # H = 200

    training_dirs = [os.sep.join([training_dir, geom, 'tensordata_normalized']) for geom in training_flow_geom]
    # select what type of data you want. see training data folder for options
    print(training_dirs)

    geneva_architecture = [5, 200, 200, 200, 40, 20, 10]
    ling_architecture = [5, 30, 30, 30, 30, 30, 30, 30, 30, 10]
    layers = geneva_architecture
    model = TensorBasedNN.TBNNModel(layersizes=layers,
                                    activation=nn.LeakyReLU(),
                                    final_layer_activation=th.tanh)

    # model.net = th.load(model_path)

    for m in model.net.modules():
        print(m)

    model.load_data(training_dirs, training_cases)
    model.select_training_data(train_ratio=0.7)
    model.train_model(learning_rate=2.5e-7, n_epochs=1000, batch_size=20)

    save_model_path = './storage/models/periodic_hills/normalized/geneva'
    th.save(model.net, os.sep.join([save_model_path, 'model.pt']))
    th.save(model.net.state_dict(), os.sep.join([save_model_path, 'state_dict.pt']))
    th.save(model.loss_vector, os.sep.join([save_model_path, 'loss_vector.th']))
    th.save(model.val_loss_vector, os.sep.join([save_model_path, 'val_loss_vector.th']))
    # model.net.reset_parameters()


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
