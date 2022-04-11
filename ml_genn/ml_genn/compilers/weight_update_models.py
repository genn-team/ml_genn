static_pulse_model = {
    "param_name_types": [("g", "scalar")],
    "sim_code":
        """
        $(addToInSyn, $(g));
        """}

static_pulse_delay_model = {
    "param_name_types": [("g", "scalar"), ("d", "uint8_t")],
    "sim_code":
        """
        $(addToInSynDelay, $(g), $(d));
        """}

signed_static_pulse_model = {
    "param_name_types": [("g", "scalar")],
    "sim_code":
        """
        $(addToInSyn, $(g));
        """,
    "event_code":
        """
        $(addToInSyn, -$(g));
        """}

signed_static_pulse_delay_model = {
    "param_name_types": [("g", "scalar"), ("d", "uint8_t")],
    "sim_code":
        """
        $(addToInSynDelay, $(g), $(d));
        """,
    "event_code":
        """
        $(addToInSynDelay, $(g), -$(d));
        """}
