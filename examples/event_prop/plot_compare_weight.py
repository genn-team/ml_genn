import numpy as np
import matplotlib.pyplot as plt
import os

def load_compare(filename):
    prefix = "COMPARE_NO_LEARN_NO_REG"
    path = "/its/home/jk421/genn_eventprop"
    
    return np.load(os.path.join(path, prefix + filename)

def plot_weights(ml_genn_filename, thomas_filename):
    in_hid_weight = np.load(ml_genn_filename)
    

    thomas_in_hid_weight = np.load(thomas_filename)
    thomas_in_hid_weight = np.reshape(thomas_in_hid_weight, in_hid_weight.shape)
    
    in_hid_weight = in_hid_weight[:,0,:,:]
    thomas_in_hid_weight = thomas_in_hid_weight[:,0,:,:]
    weight_diff = np.abs(in_hid_weight - thomas_in_hid_weight)
    min_weight = min(np.amin(in_hid_weight), np.amin(thomas_in_hid_weight))
    max_weight = max(np.amax(in_hid_weight), np.amax(thomas_in_hid_weight))

    fig, axes = plt.subplots(3, 6, sharex="col", sharey="row")
    for i in range(6):
        axes[0,i].set_title(f"Trial {i}")
        axes[0,i].imshow(in_hid_weight[i], aspect="auto", vmin=min_weight, vmax=max_weight)
        axes[1,i].imshow(thomas_in_hid_weight[i], aspect="auto", vmin=min_weight, vmax=max_weight)
        axes[2,i].imshow(weight_diff[i], aspect="auto")

    axes[0,0].set_ylabel("mlGeNN")
    axes[1,0].set_ylabel("Thomas")
    axes[2,0].set_ylabel("Difference")
    return fig, axes

in_hid_fig, in_hid_axes = plot_weights("in_hid_g_hack.npy", "COMPARE_HACK_FABS_win_to_hid.npy")
in_hid_fig.suptitle("Input to hidden weights")

hid_out_fig, hid_out_axes = plot_weights("hid_out_g_hack.npy", "COMPARE_HACK_FABS_whid_to_out.npy")
hid_out_fig.suptitle("Hidden to output weights")
plt.show()
