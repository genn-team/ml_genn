import matplotlib.pyplot as plt
import numpy as np
import os

def load_compare(title):
    prefix = "COMPARE_NO_LEARN_REG_"
    path = "/its/home/jk421/genn_eventprop"
    
    return np.load(os.path.join(path, prefix + title + ".npy"))
    
in_spike_ids = np.load("in_spike_ids_hack.npz")
in_spike_times = np.load("in_spike_times_hack.npz")
hidden_spike_times = np.load("hidden_spike_times_hack.npz")
hidden_spike_ids = np.load("hidden_spike_ids_hack.npz")
hidden_lambda_i = np.load("hidden_lambda_i_hack.npy")
hidden_lambda_v = np.load("hidden_lambda_v_hack.npy")
hidden_spike_time = np.load("hidden_spike_count_hack.npy")
hidden_spike_time_back = np.load("hidden_spike_count_back_hack.npy") / 32
out_v = np.load("out_v_hack.npy")
hidden_v = np.load("hidden_v_hack.npy")
out_lambda_i = np.load("out_lambda_i_hack.npy")
out_lambda_v = np.load("out_lambda_v_hack.npy")

thomas_in_spike_ids = load_compare("input_spike_ID")
thomas_in_spike_times = load_compare("input_spike_t") + 1.0
thomas_hidden0_spike_ids = load_compare("hidden0_spike_ID")
thomas_hidden0_spike_times = load_compare("hidden0_spike_t") + 1.0
thomas_hidden_lambda_v = load_compare("lambda_Vhidden0")
thomas_hidden_lambda_i = load_compare("lambda_Ihidden0")
thomas_hidden_spike_time = load_compare("new_sNSumhidden0")
thomas_hidden_spike_time_back = load_compare("sNSumhidden0")
thomas_out_v = load_compare("Voutput")
thomas_hidden_v = load_compare("Vhidden0")
thomas_out_lambda_v = load_compare("lambda_Voutput")
thomas_out_lambda_i = load_compare("lambda_Ioutput")

num_trials = len(in_spike_ids)
assert len(in_spike_times) == num_trials
assert len(hidden_spike_times) == num_trials
assert len(hidden_spike_ids) == num_trials
assert out_v.shape[0] == num_trials
assert out_v.shape[1] == 1000
assert out_v.shape[2] == 20
assert out_lambda_i.shape[0] == num_trials
assert out_lambda_i.shape[1] == 1000
assert out_lambda_i.shape[2] == 20
assert out_lambda_v.shape[0] == num_trials
assert out_lambda_v.shape[1] == 1000
assert out_lambda_v.shape[2] == 20
assert (thomas_out_v.shape[0] // 1000) == num_trials

num_trials = 3
spikes_fig, spikes_axes = plt.subplots(2, num_trials, sharex="col", sharey="row")
spikes_fig.suptitle("Spikes")

hid_fig, hid_axes = plt.subplots(10, num_trials, sharex="col", sharey="row")
hid_fig.suptitle("Hidden layer lambda")

hid_v_fig, hid_v_axes = plt.subplots(10, num_trials, sharex="col", sharey="row")
hid_v_fig.suptitle("Hidden layer voltages")

hid_spike_count_fig, hid_spike_count_axes = plt.subplots(10, num_trials, sharex="col", sharey="row")
hid_spike_count_fig.suptitle("Hidden layer spike counts")


out_fig, out_axes = plt.subplots(10, num_trials, sharex="col", sharey="row")
out_fig.suptitle("Output layer lambda")

out_v_fig, out_v_axes = plt.subplots(10, num_trials, sharex="col", sharey="row")
out_v_fig.suptitle("Output layer voltages")

for i in range(num_trials):
    thomas_start_idx = 1000 * i
    thomas_end_idx = thomas_start_idx + 1000
    thomas_start_t = i * 32 * 1000
    thomas_end_t = thomas_start_t + 1000
    
    thomas_in_spike_mask = (thomas_in_spike_times >= thomas_start_t) & (thomas_in_spike_times < thomas_end_t)
    ml_genn_spike_actor = spikes_axes[0, i].scatter(in_spike_times[f"arr_{i}"], in_spike_ids[f"arr_{i}"], s=1, alpha=0.5)
    thomas_spike_actor = spikes_axes[0, i].scatter(thomas_in_spike_times[thomas_in_spike_mask] - thomas_start_t, 
                                                   thomas_in_spike_ids[thomas_in_spike_mask], s=1, alpha=0.5)

    thomas_hid_spike_mask = (thomas_hidden0_spike_times >= thomas_start_t) & (thomas_hidden0_spike_times < thomas_end_t)
    spikes_axes[1,i].scatter(hidden_spike_times[f"arr_{i}"], hidden_spike_ids[f"arr_{i}"], s=1, alpha=0.5)
    spikes_axes[1,i].scatter(thomas_hidden0_spike_times[thomas_hid_spike_mask] - thomas_start_t, 
                             thomas_hidden0_spike_ids[thomas_hid_spike_mask], s=1, alpha=0.5)
    
    # Loop over hidden neurons
    for j in range(10):
        assert np.all(np.isclose(hidden_spike_time_back[i,0,j], hidden_spike_time_back[i,:,j]))
        assert np.all(np.isclose(thomas_hidden_spike_time_back[thomas_start_idx,0, j], thomas_hidden_spike_time_back[thomas_start_idx:thomas_end_idx,0, j]))

        # Plot mlGeNN values
        ml_genn_hid_v_actor = hid_v_axes[j, i].plot(hidden_v[i,:,j], alpha=0.5)[0]
        hid_lambda_v_actor = hid_axes[j,i].plot(hidden_lambda_v[i,:,j], alpha=0.5)[0]
        hid_lambda_i_actor = hid_axes[j,i].plot(hidden_lambda_i[i,:,j], alpha=0.5)[0]
        
        hid_spike_count_axes[j,i].plot(hidden_spike_time[i,:,j], alpha=0.5)
        
        spike_count_axis = hid_axes[j,i].twinx()
        hid_spike_count_actor = spike_count_axis.axhline(hidden_spike_time_back[i,0,j], alpha=0.5)
        
        # Plot Thomas
        thomas_hid_v_actor = hid_v_axes[j, i].plot(thomas_hidden_v[thomas_start_idx:thomas_end_idx,0, j],alpha=0.5)[0]
        hid_axes[j,i].plot(thomas_hidden_lambda_v[thomas_start_idx:thomas_end_idx,0, j],
                           linestyle="--", alpha=0.5, color=hid_lambda_v_actor.get_color())
        hid_axes[j,i].plot(thomas_hidden_lambda_i[thomas_start_idx:thomas_end_idx,0, j],
                           linestyle="--", alpha=0.5, color=hid_lambda_i_actor.get_color())
        
        hid_spike_count_axes[j,i].plot(thomas_hidden_spike_time[thomas_start_idx:thomas_end_idx,0, j])
        
        spike_count_axis.axhline(thomas_hidden_spike_time_back[thomas_start_idx,0, j],
                                 alpha=0.5, color=hid_spike_count_actor.get_color(), linestyle="--")
        
    for j in range(10):
        # Plot mlGeNN values
        ml_genn_out_v_actor = out_v_axes[j, i].plot(out_v[i,:,j])[0]
        out_lambda_v_actor = out_axes[j,i].plot(out_lambda_v[i,:,j + 10], alpha=0.5)[0]
        out_lambda_i_actor = out_axes[j,i].plot(out_lambda_i[i,:,j + 10], alpha=0.5)[0]
        
        # Plot Thomas
        thomas_out_v_actor = out_v_axes[j, i].plot(thomas_out_v[thomas_start_idx:thomas_end_idx,0, j], alpha=0.5)[0]
        out_axes[j,i].plot(thomas_out_lambda_v[thomas_start_idx:thomas_end_idx,0, j + 10],
                           linestyle="--", alpha=0.5, color=out_lambda_v_actor.get_color())
        out_axes[j,i].plot(thomas_out_lambda_i[thomas_start_idx:thomas_end_idx,0, j + 10],
                           linestyle="--", alpha=0.5, color=out_lambda_i_actor.get_color())

    #out_secondary_axes[0].get_shared_y_axes().join(*out_secondary_axes[1:])

    spikes_axes[0,i].set_title(f"Trial {i}")
    hid_axes[0,i].set_title(f"Trial {i}")
    out_axes[0,i].set_title(f"Trial {i}")
    spikes_axes[-1,i].set_xlabel("t [ms]")
    hid_axes[-1,i].set_xlabel("t [ms]")
    out_axes[-1,i].set_xlabel("t [ms]")

#axes[0,-1].legend()
#axes[2,-1].legend()
spikes_axes[0,0].set_ylabel("Input spike ID")
spikes_axes[1,0].set_ylabel("Hidden spike ID")

spikes_fig.legend([ml_genn_spike_actor, thomas_spike_actor],
                  ["mlGeNN", "Thomas"], loc="lower center", ncol=2)
hid_fig.legend([hid_lambda_v_actor, hid_lambda_i_actor, hid_spike_count_actor],
               ["LambdaV", "LambdaI", "Spike count"], loc="lower center", ncol=3)
hid_v_fig.legend([ml_genn_hid_v_actor, thomas_hid_v_actor],
                 ["mlGeNN", "Thomas"], loc="lower center", ncol=2)
out_fig.legend([out_lambda_v_actor, out_lambda_i_actor],
               ["LambdaV", "LambdaI"], loc="lower center", ncol=2)
out_v_fig.legend([ml_genn_out_v_actor, thomas_out_v_actor],
                 ["mlGeNN", "Thomas"], loc="lower center", ncol=2)

hid_spike_count_fig.tight_layout(pad=0)
spikes_fig.tight_layout(pad=0)
hid_fig.tight_layout(pad=0)
out_fig.tight_layout(pad=0)
out_v_fig.tight_layout(pad=0)
hid_v_fig.tight_layout(pad=0)

plt.show()
