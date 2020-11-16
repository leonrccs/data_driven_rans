from scripts import TensorBasedNN
import torch as th
import os

if __name__ == '__main__':
    training_dir = '/home/leonriccius/Documents/Fluid_Data/training_data/periodic_hills/tensordata'
    training_cases = ['700', '1400', '5600'] #, '10595']
    model_path = './notebooks/trained_models/3cases/state_dict.pt'

    d_in, d_out = 5, 10
    H = 200

    NN = TensorBasedNN.TBNNModel(d_in, H, d_out)

    NN.model.load_state_dict(th.load(model_path))
    NN.model.eval()

    NN.load_data(training_dir, training_cases)
    NN.select_training_data(train_ratio=0.7)

    # NN.train_model(learning_rate=1e-6, n_epochs=500, batch_size=20)

    new_model_path = './notebooks/trained_models/3cases'
    th.save(NN.model, os.sep.join([new_model_path, 'model.pt']))
    # th.save(NN.model.state_dict(), os.sep.join([new_model_path, 'state_dict.pt']))
    # th.save(NN.loss_vector, os.sep.join([new_model_path, 'loss_vector.th']))
    # th.save(NN.val_loss_vector, os.sep.join([new_model_path, 'val_loss_vector.th']))
    # NN.model.reset_parameters()
