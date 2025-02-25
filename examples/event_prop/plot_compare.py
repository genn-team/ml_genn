import matplotlib.pyplot as plt
import numpy as np
import os

tag_1 = "_master"
tag_1_in_spike_ids = np.load(f"in_spike_ids{tag_1}.npz")
tag_1_in_spike_times = np.load(f"in_spike_times{tag_1}.npz")
tag_1_hidden_spike_times = np.load(f"hidden_spike_times{tag_1}.npz")
tag_1_hidden_spike_ids = np.load(f"hidden_spike_ids{tag_1}.npz")
tag_1_hidden_lambda_i = np.load(f"hidden_lambda_i{tag_1}.npy")
tag_1_hidden_lambda_v = np.load(f"hidden_lambda_v{tag_1}.npy")
tag_1_hidden_spike_time = np.load(f"hidden_spike_count{tag_1}.npy")
tag_1_hidden_spike_time_back = np.load(f"hidden_spike_count_back{tag_1}.npy") / 32
tag_1_out_v = np.load(f"out_v{tag_1}.npy")
tag_1_hidden_v = np.load(f"hidden_v{tag_1}.npy")
tag_1_out_lambda_i = np.load(f"out_lambda_i{tag_1}.npy")
tag_1_out_lambda_v = np.load(f"out_lambda_v{tag_1}.npy")

tag_2 = "_auto"
tag_2_in_spike_ids = np.load(f"in_spike_ids{tag_2}.npz")
tag_2_in_spike_times = np.load(f"in_spike_times{tag_2}.npz")
tag_2_hidden_spike_times = np.load(f"hidden_spike_times{tag_2}.npz")
tag_2_hidden_spike_ids = np.load(f"hidden_spike_ids{tag_2}.npz")
#tag_2_hidden_lambda_i = np.load(f"hidden_lambda_i{tag_2}.npy")
tag_2_hidden_lambda_v = np.load(f"hidden_lambda_v{tag_2}.npy")
tag_2_hidden_spike_time = np.load(f"hidden_spike_count{tag_2}.npy")
tag_2_hidden_spike_time_back = np.load(f"hidden_spike_count_back{tag_2}.npy") / 32
tag_2_out_v = np.load(f"out_v{tag_2}.npy")
tag_2_hidden_v = np.load(f"hidden_v{tag_2}.npy")
#tag_2_out_lambda_i = np.load(f"out_lambda_i{tag_2}.npy")
tag_2_out_lambda_v = np.load(f"out_lambda_v{tag_2}.npy")

num_trials = len(tag_1_in_spike_ids)
assert len(tag_1_in_spike_times) == num_trials
assert len(tag_1_hidden_spike_times) == num_trials
assert len(tag_1_hidden_spike_ids) == num_trials
assert tag_1_out_v.shape[0] == num_trials
assert tag_1_out_v.shape[1] == 1000
assert tag_1_out_v.shape[2] == 20
assert tag_1_out_lambda_i.shape[0] == num_trials
assert tag_1_out_lambda_i.shape[1] == 1000
assert tag_1_out_lambda_i.shape[2] == 20
assert tag_1_out_lambda_v.shape[0] == num_trials
assert tag_1_out_lambda_v.shape[1] == 1000
assert tag_1_out_lambda_v.shape[2] == 20

assert len(tag_2_in_spike_times) == num_trials
assert len(tag_2_hidden_spike_times) == num_trials
assert len(tag_2_hidden_spike_ids) == num_trials
assert tag_2_out_v.shape[0] == num_trials
assert tag_2_out_v.shape[1] == 1000
assert tag_2_out_v.shape[2] == 20
#assert tag_2_out_lambda_i.shape[0] == num_trials
#assert tag_2_out_lambda_i.shape[1] == 1000
#assert tag_2_out_lambda_i.shape[2] == 20
assert tag_2_out_lambda_v.shape[0] == num_trials
assert tag_2_out_lambda_v.shape[1] == 1000
assert tag_2_out_lambda_v.shape[2] == 20

num_trials = 3
vis_num_hidden = 3
vis_num_out = 3

spikes_fig, spikes_axes = plt.subplots(2, num_trials, sharex="col", sharey="row")
spikes_fig.suptitle("Spikes")

hid_fig, hid_axes = plt.subplots(vis_num_hidden, num_trials, sharex="col", sharey="row")
hid_fig.suptitle("Hidden layer lambda")

hid_v_fig, hid_v_axes = plt.subplots(vis_num_hidden, num_trials, sharex="col", sharey="row")
hid_v_fig.suptitle("Hidden layer voltages")

hid_spike_count_fig, hid_spike_count_axes = plt.subplots(vis_num_hidden, num_trials, sharex="col", sharey="row")
hid_spike_count_fig.suptitle("Hidden layer spike counts")


out_fig, out_axes = plt.subplots(vis_num_out, num_trials, sharex="col", sharey="row")
out_fig.suptitle("Output layer lambda")

out_v_fig, out_v_axes = plt.subplots(vis_num_out, num_trials, sharex="col", sharey="row")
out_v_fig.suptitle("Output layer voltages")

for i in range(num_trials):
    tag_1_spike_actor = spikes_axes[0, i].scatter(tag_1_in_spike_times[f"arr_{i}"], tag_1_in_spike_ids[f"arr_{i}"], s=1, alpha=0.5)
    tag_2_spike_actor = spikes_axes[0, i].scatter(tag_2_in_spike_times[f"arr_{i}"], tag_2_in_spike_ids[f"arr_{i}"], s=1, alpha=0.5)

    spikes_axes[1,i].scatter(tag_1_hidden_spike_times[f"arr_{i}"], tag_1_hidden_spike_ids[f"arr_{i}"], s=1, alpha=0.5)
    spikes_axes[1,i].scatter(tag_2_hidden_spike_times[f"arr_{i}"], tag_2_hidden_spike_ids[f"arr_{i}"], s=1, alpha=0.5)
    
    # Loop over hidden neurons
    for j in range(vis_num_hidden):
        assert np.all(np.isclose(tag_1_hidden_spike_time_back[i,0,j], tag_1_hidden_spike_time_back[i,:,j]))
        assert np.all(np.isclose(tag_2_hidden_spike_time_back[i,0,j], tag_2_hidden_spike_time_back[i,:,j]))

        # Plot tag 1 values
        tag_1_hid_v_actor = hid_v_axes[j, i].plot(tag_1_hidden_v[i,:,j], alpha=0.5)[0]
        hid_lambda_v_actor = hid_axes[j,i].plot(tag_1_hidden_lambda_v[i,:,j] * 20.0, alpha=0.5)[0]
        #hid_lambda_i_actor = hid_axes[j,i].plot(tag_1_hidden_lambda_i[i,:,j] * 5.0, alpha=0.5)[0]
        
        hid_spike_count_axes[j,i].plot(tag_1_hidden_spike_time[i,:,j], alpha=0.5)
        
        #spike_count_axis = hid_axes[j,i].twinx()
        #hid_spike_count_actor = spike_count_axis.axhline(tag_1_hidden_spike_time_back[i,0,j], alpha=0.5)
        
        # Plot tag 2 values
        tag_2_hid_v_actor = hid_v_axes[j, i].plot(tag_2_hidden_v[i,:,j], alpha=0.5)[0]
        hid_axes[j,i].plot(tag_2_hidden_lambda_v[i,:,j], linestyle="--", alpha=0.5,
                           color=hid_lambda_v_actor.get_color())
        #hid_axes[j,i].plot(tag_2_hidden_lambda_i[i,:,j], linestyle="--",
        #                   alpha=0.5, color=hid_lambda_i_actor.get_color())
        
        hid_spike_count_axes[j,i].plot(tag_2_hidden_spike_time[i,:,j], alpha=0.5)
   
        #spike_count_axis.axhline(tag_2_hidden_spike_time_back[i,0,j], alpha=0.5,
        #                         color=hid_spike_count_actor.get_color(), linestyle="--")
        
    for j in range(vis_num_out):
        # Plot tag 1 values
        tag_1_out_v_actor = out_v_axes[j, i].plot(tag_1_out_v[i,:,j])[0]
        out_lambda_v_actor = out_axes[j,i].plot(tag_1_out_lambda_v[i,:,j + 10] * 20.0, alpha=0.5)[0]
        #out_lambda_i_actor = out_axes[j,i].plot(tag_1_out_lambda_i[i,:,j + 10] * 5.0, alpha=0.5)[0]
        
        # Plot tag 2 values
        tag_2_out_v_actor = out_v_axes[j, i].plot(tag_2_out_v[i,:,j], alpha=0.5)[0]
        out_axes[j,i].plot(tag_2_out_lambda_v[i,:,j + 10], linestyle="--", 
                           alpha=0.5, color=out_lambda_v_actor.get_color())
        #out_axes[j,i].plot(tag_2_out_lambda_i[i,:,j + 10], linestyle="--", 
        #                   alpha=0.5, color=out_lambda_i_actor.get_color())[0]
       
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

spikes_fig.legend([tag_1_spike_actor, tag_2_spike_actor],
                  [tag_1[1:], tag_2[1:]], loc="lower center", ncol=2)
#hid_fig.legend([hid_lambda_v_actor, hid_lambda_i_actor, hid_spike_count_actor],
#               ["LambdaV", "LambdaI", "Spike count"], loc="lower center", ncol=3)
hid_v_fig.legend([tag_1_hid_v_actor, tag_2_hid_v_actor],
                 [tag_1[1:], tag_2[1:]], loc="lower center", ncol=2)
#out_fig.legend([out_lambda_v_actor, out_lambda_i_actor],
#               ["LambdaV", "LambdaI"], loc="lower center", ncol=2)
out_v_fig.legend([tag_1_out_v_actor, tag_2_out_v_actor],
                 [tag_1[1:], tag_2[1:]], loc="lower center", ncol=2)

hid_spike_count_fig.tight_layout(pad=0)
spikes_fig.tight_layout(pad=0)
hid_fig.tight_layout(pad=0)
out_fig.tight_layout(pad=0)
out_v_fig.tight_layout(pad=0)
hid_v_fig.tight_layout(pad=0)

plt.show()
