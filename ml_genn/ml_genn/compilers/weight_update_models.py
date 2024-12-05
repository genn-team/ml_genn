static_pulse_model = {
    "params": [("g", "scalar")],
    "pre_spike_syn_code":
        """
        addToPost(g);
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


def get_static_pulse_delay_model(delay_type):
    return {"params": [("g", "scalar"), 
                       ("d", delay_type)],
            "pre_spike_syn_code":
                """
                addToPostDelay(g, d);
                """}

def get_signed_static_pulse_delay_model(delay_type):
    return {"params": [("g", "scalar"), 
                       ("d", delay_type)],
            "pre_spike_syn_code":
                """
                addToPostDelay(g, d);
                """,
            "pre_event_syn_code":
                """
                addToPostDelay(g, -d);
                """}