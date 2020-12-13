from pygenn.genn_model import create_custom_weight_update_class
from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY

signed_static_pulse = create_custom_weight_update_class(
    'signed_static_pulse',
    var_name_types=[("g", "scalar", VarAccess_READ_ONLY)],
    sim_code='''
    $(addToInSyn, $(g));
    ''',
    event_code='''
    $(addToInSyn, -$(g));
    ''',
    event_threshold_condition_code='''
    $(input_pre) < 0.0 && spike
    '''
)
