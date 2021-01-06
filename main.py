from scripts import TensorBasedNN
import torch as th
import torch.nn as nn
import os

if __name__ == '__main__':
    training_dir = '/home/leonriccius/Documents/Fluid_Data/training_data/periodic_hills/tensordata_rans_grid'
    training_cases = ['700', '1400', '5600'] #, '10595']
    # model_path = './notebooks/trained_models/3cases/state_dict.pt'
    # model_path = './storage/models/periodic_hills/1000epochs/model.pt'

    # d_in, d_out = 5, 10
    # H = 200

    layers = [5, 30, 30, 30, 30, 30, 30, 30, 30, 10]
    # geneva architecture [5, 200, 200, 200, 40, 20, 10]
    # ling-architecture [5, 30, 30, 30, 30, 30, 30, 30, 30, 10]
    model = TensorBasedNN.TBNNModel(layersizes=layers,
                                    activation=nn.LeakyReLU(),
                                    final_layer_activation=th.tanh)

    # model.net = th.load(model_path)

    for m in model.net.modules():
        print(m)

    model.load_data(training_dir, training_cases)
    model.select_training_data(train_ratio=0.7)
    model.train_model(learning_rate=2.5e-7, n_epochs=1000, batch_size=20)

    new_model_path = './storage/models/periodic_hills/whole_domain'
    th.save(model.net, os.sep.join([new_model_path, 'model.pt']))
    th.save(model.net.state_dict(), os.sep.join([new_model_path, 'state_dict.pt']))
    th.save(model.loss_vector, os.sep.join([new_model_path, 'loss_vector.th']))
    th.save(model.val_loss_vector, os.sep.join([new_model_path, 'val_loss_vector.th']))
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
