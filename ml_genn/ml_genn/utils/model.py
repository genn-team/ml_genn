from collections import namedtuple

CustomUpdateModel = namedtuple("CustomUpdateModel", ["model", "param_vals", 
                                                     "var_vals", "var_refs"])
NeuronModel = namedtuple("NeuronModel", ["model", "param_vals", "var_vals"])
SynapseModel = namedtuple("SynapseModel", ["model", "param_vals", "var_vals"])