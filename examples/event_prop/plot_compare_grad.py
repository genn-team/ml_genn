import numpy as np
import matplotlib.pyplot as plt
import os

def load_compare(title):
    prefix = "COMPARE_NO_LEARN_REG_"
    path = "/its/home/jk421/genn_eventprop"
    
    return np.load(os.path.join(path, prefix + title + ".npy"))

def plot_grads(ml_genn_filename, thomas_title):
    grad = np.load(ml_genn_filename)
    grad = np.reshape(grad, grad.shape[:2] + (-1,))

    thomas_grad = load_compare(thomas_title)
    thomas_grad = np.reshape(thomas_grad, grad.shape)
    
    num_viz = 20
    synapse_inds = np.random.choice(grad.shape[-1], num_viz)
    fig, axes = plt.subplots(num_viz, 6, sharex="col", sharey="row")
    for i in range(6):
        axes[0,i].set_title(f"Trial {i}")
        
        for j in range(num_viz):
            ml_genn_actor = axes[j,i].plot(grad[i,:,synapse_inds[j]], alpha=0.5)[0]
            thomas_actor = axes[j,i].plot(thomas_grad[i,:,synapse_inds[j]], alpha=0.5)[0]

    fig.legend([ml_genn_actor, thomas_actor], ["mlGeNN", "Thomas"],
               ncol=2, loc="lower center")
    fig.tight_layout(pad=0)
   
    return fig, axes

in_hid_fig, in_hid_axes = plot_grads("in_hid_grad_hack.npy", "dwin_to_hid")
in_hid_fig.suptitle("Input to hidden gradients")

hid_out_fig, hid_out_axes = plot_grads("hid_out_grad_hack.npy", "dwhid_to_out")
hid_out_fig.suptitle("Hidden to output gradients")
plt.show()
