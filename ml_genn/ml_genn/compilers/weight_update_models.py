static_pulse_model = {
    "params": [("g", "scalar")],
    "pre_spike_syn_code":
        """
        addToPost(g);
        """}

static_pulse_delay_model = {
    "params": [("g", "scalar"), ("d", "uint8_t")],
    "pre_spike_syn_code":
        """
        addToPostDelay(g, d);
        """}

signed_static_pulse_model = {
    "params": [("g", "scalar")],
    "pre_spike_syn_code":
        """
        addToPost(g);
        """,
    "pre_event_syn_code":
        """
        addToPost(-g);
        """}

signed_static_pulse_delay_model = {
    "params": [("g", "scalar"), ("d", "uint8_t")],
    "pre_spike_syn_code":
        """
        addToPostDelay(g, d);
        """,
    "pre_event_syn_code":
        """
        addToPostDelay(g, -d);
        """}
