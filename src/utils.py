import h5py
import torch


def load_h5_weights(model, weights_path):
    with h5py.File(weights_path, 'r') as f:
        for layer in model.layers:
            if layer.weights:
                weights = []
                for weight in layer.weights:
                    weight_name = weight.name
                    if weight_name in f:
                        weight_value = torch.tensor(f[weight_name][:])
                        weights.append(weight_value)
                if weights:
                    layer.set_weights(weights)
    return model