from collections import namedtuple

CustomUpdateModel = namedtuple("CustomUpdateModel", 
                               ["model", "param_vals", "var_vals", 
                                "var_refs", "egp_vals"],
                               defaults=[{}, {}, {}, {}])
NeuronModel = namedtuple("NeuronModel", 
                         ["model", "param_vals", "var_vals", "egp_vals"], 
                         defaults=[{}, {}, {}])
SynapseModel = namedtuple("SynapseModel", 
                          ["model", "param_vals", "var_vals", "egp_vals"],
                          defaults=[{}, {}, {}])
