from os import mkdir
import numpy as np


def save_model(model, path):
    mkdir(path)

    with open(path + '/name', 'w') as out_file:
        out_file.write(model.name + '\n')

    with open(path + '/layer_names', 'w') as out_file:
        for i in range(len(model.layer_names)):
            out_file.write(model.layer_names[i] + '\n')

    for i in range(len(model.weight_vals)):
        np.savetxt(path + '/weight_vals_' + model.layer_names[i], model.weight_vals[i])

    for i in range(len(model.weight_conn)):
        np.savetxt(path + '/weight_conn_' + model.layer_names[i], model.weight_conn[i])

    for i in range(len(model.thresholds)):
        np.savetxt(path + '/threshold_' + model.layer_names[i], model.thresholds[i])


def load_model(path):
    with open(path + '/name', 'r') as in_file:
        model.name = in_file.read().splitlines()

    with open(path + '/layer_names', 'r') as in_file:
        for i in range(len(model.layer_names)):
            model.layer_names[i] = in_file.read().splitlines()

    for i in range(len(model.weight_vals)):
        model.weight_vals[i]= np.loadtxt(path + '/weight_vals_' + model.layer_names[i])

    for i in range(len(model.weight_conn)):
        model.weight_conn[i]= np.loadtxt(path + '/weight_conn_' + model.layer_names[i])

    for i in range(len(model.thresholds)):
        model.thresholds[i]= np.loadtxt(path + '/threshold_' + model.layer_names[i])
