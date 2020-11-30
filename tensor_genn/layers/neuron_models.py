from pygenn.genn_model import create_custom_neuron_class
from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY

# === IF neuron class ===
if_model = create_custom_neuron_class(
    'if',
    var_name_types=[('Vmem', 'scalar'), ('nSpk', 'unsigned int')],
    extra_global_params=[('Vthr', 'scalar')],
    sim_code='''
    if ($(t) == 0.0) {
        // Reset state at t = 0
        $(Isyn) = 0.0;
        $(Vmem) = 0.0;
        $(nSpk) = 0;
    }
    $(Vmem) += $(Isyn) * DT;
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


# === Spike input neuron class ===
spike_input_model = create_custom_neuron_class(
    'spike_input',
    var_name_types=[('input', 'scalar', VarAccess_READ_ONLY)],
    threshold_condition_code='''
    $(input)
    ''',
    is_auto_refractory_required=False,
)


# === Poisson input neuron class ===
poisson_input_model = create_custom_neuron_class(
    'poisson_input',
    var_name_types=[('input', 'scalar', VarAccess_READ_ONLY)],
    sim_code='''
    const scalar u = $(gennrand_uniform);
    ''',
    threshold_condition_code='''
    $(input) > 0 && u >= exp(-$(input) * DT)
    ''',
    is_auto_refractory_required=False,
)


# === IF input neuron class ===
if_input_model = create_custom_neuron_class(
    'if_input',
    var_name_types=[('input', 'scalar', VarAccess_READ_ONLY), ('Vmem', 'scalar')],
    sim_code='''
    if ($(t) == 0.0) {
        // Reset state at t = 0
        $(Vmem) = 0.0;
    }
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
