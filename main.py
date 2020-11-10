from scripts import TensorBasedNN
import torch as th

if __name__ == '__main__':
    training_dir = '/home/leonriccius/Documents/Fluid_Data/training_data/periodic_hills/tensordata'
    training_cases = ['700', '1400', '5600', '10595']

    d_in, d_out = 5, 10
    H = 200

    NN = TensorBasedNN.TBNNModel(d_in, H, d_out)

    NN.load_data(training_dir, training_cases)
    NN.select_training_data(train_ratio=0.1)

    NN.train_model(learning_rate=1e-5, n_epochs=40, batch_size=20)

    NN.model.reset_parameters()

    NN.train_model(learning_rate=1e-5, n_epochs=40, batch_size=20)
