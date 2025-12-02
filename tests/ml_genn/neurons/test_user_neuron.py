from ml_genn.neurons import UserNeuron

def test_conflicting_var_name():
    lif_beta = UserNeuron(vars={"beta": ("(-beta + Isyn) / tau_mem", "0.0")},
                          threshold="beta - 1.0",
                          output_var_name="beta",
                          param_vals={"tau_mem": 20.0},
                          var_vals={"beta": 0.0})
    lif_beta.get_model(population=None, dt=1.0, batch_size=1)

def test_izhikevich():
    izhikevich = UserNeuron(vars={"V": ("(0.04 * V**2) + (5 * V) + 140 - U + Isyn", "c"),
                                  "U": ("a * ((b * V) - U)", "U + d")},
                          threshold="30.0",
                          output_var_name="V",
                          param_vals={"a": 0.02, "b": 0.02, "c": -65.0, "d": 8.0},
                          var_vals={"V": -65.0, "U": -20.0})
    izhikevich.get_model(population=None, dt=1.0, batch_size=1)

def test_adexp():
    adexp = UserNeuron(vars={"V": ("(1.0 / c) * ((-gL * (V - eL)) + (gL * deltaT * exp((V - vThresh) / deltaT)) + Isyn - W)", "vReset"),
                             "W": ("(1.0 / tauW) * ((a * (V - eL)) - W)", "W + b")},
                          threshold="-40.0",
                          output_var_name="V",
                          param_vals={"c": 281.0, "gL": 30.0, "eL": -70.6, 
                                      "deltaT": 2.0, "vThresh": -50.4,
                                      "vSpike": 10.0, "vReset": -70.6,
                                      "tauW": 144.0, "a": 4.0, "b": 0.0805},
                          var_vals={"V": -70.6, "W": 0.0})
    adexp.get_model(population=None, dt=1.0, batch_size=1)