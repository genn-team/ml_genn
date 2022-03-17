from collections import namedtuple

ConnectivitySnippet = namedtuple("ConnectivitySnippet", 
                                 ["snippet", "matrix_type", 
                                  "weight", "delay"])

InitializerSnippet = namedtuple("InitializerSnippet", 
                                ["snippet", "param_vals", "egp_vals"])
