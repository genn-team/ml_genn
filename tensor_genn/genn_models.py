from pygenn.genn_model import create_custom_neuron_class

# === IF neuron class ===
if_model = create_custom_neuron_class(
    'if_model',
    var_name_types=[('Vmem', 'scalar'), ('Vmem_peak', 'scalar'), ('nSpk', 'unsigned int')],
    extra_global_params=[('Vthr', 'scalar')],
    sim_code='''
    $(Vmem) += $(Isyn) * DT;
    $(Vmem_peak) = $(Vmem);
    ''',
    threshold_condition_code='''
    $(Vmem) >= $(Vthr)
    ''',
    reset_code='''
    $(Vmem) = 0.0;
    $(nSpk) += 1;
    ''',
    is_auto_refractory_required=False,
)

if_init = {
    'Vmem': 0.0,
    'Vmem_peak': 0.0,
    'nSpk': 0,
}

# === IF input neuron class ===
if_input_model = create_custom_neuron_class(
    'if_input_model',
    var_name_types=[('input', 'scalar'), ('Vmem', 'scalar')],
    sim_code='''
    $(Vmem) += $(input) * DT;
    ''',
    threshold_condition_code='''
    $(Vmem) >= 1.0
    ''',
    reset_code='''
    $(Vmem) = 0.0;
    ''',
    is_auto_refractory_required=False,
)

if_input_init = {
    'input': 0.0,
    'Vmem': 0.0,
}

# === Poisson input neuron class ===
poisson_input_model = create_custom_neuron_class(
    'poisson_input_model',
    var_name_types=[('input', 'scalar')],
    param_names=['rate_factor'],
    threshold_condition_code='''
    $(gennrand_uniform) >= exp(-$(input) * $(rate_factor) * DT)
    ''',
    is_auto_refractory_required=False,
)

poisson_input_init = {
    'input': 0.0,
}

# === Spike input neuron class ===
spike_input_model = create_custom_neuron_class(
    'spike_input_model',
    var_name_types=[('input', 'scalar')],
    threshold_condition_code='''
    $(input)
    ''',
    is_auto_refractory_required=False,
)

spike_input_init = {
    'input': 0.0,
}
