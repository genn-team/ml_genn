static_pulse_model = {
    "param_name_types": [("g", "scalar")],
    "sim_code":
        """
        addToPost(g);
        """}

static_pulse_delay_model = {
    "param_name_types": [("g", "scalar"), ("d", "uint8_t")],
    "sim_code":
        """
        addToPostDelay(g, d);
        """}

signed_static_pulse_model = {
    "param_name_types": [("g", "scalar")],
    "sim_code":
        """
        addToPost(g);
        """,
    "event_code":
        """
        addToPost(-g);
        """}

signed_static_pulse_delay_model = {
    "param_name_types": [("g", "scalar"), ("d", "uint8_t")],
    "sim_code":
        """
        addToPostDelay(g, d);
        """,
    "event_code":
        """
        addToPostDelay(g, -d);
        """}
