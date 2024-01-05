import numpy as np

from ml_genn.initializers import Uniform
from ml_genn.utils.model import (NeuronModel, WeightUpdateModel)

from pygenn import VarAccess

from pytest import raises

def test_process():
    nm = NeuronModel({"params": [("P1", "scalar"), ("P2", "scalar"),
                                 ("P3", "scalar")],
                      "vars": [("V", "scalar")]},
                     "V", {"P1": 3.0, "P2": np.arange(4), "P3": Uniform()},
                     {"V": 1.0})
    
    model_copy, constant_param_vals, _, var_vals, _, _ = nm.process()
    
    assert "P1" in constant_param_vals
    assert "P2" in var_vals
    assert "P3" in var_vals
    assert "V" in var_vals

def test_reset_vars():
    nm = NeuronModel({"vars": [("V", "scalar"),
                               ("VRW", "scalar", VarAccess.READ_WRITE),
                               ("VRO", "int", VarAccess.READ_ONLY)]},
                    "V", {}, {"V": 1.0, "VRW": 2.0, "VRO": 3.0})
    reset_vars = nm.reset_vars
    assert len(reset_vars) == 2
    assert reset_vars[0] == ("V", "scalar", 1.0)
    assert reset_vars[1] == ("VRW", "scalar", 2.0)

def test_weight_update_pre_post_reset_vars():
    wum = WeightUpdateModel(
        {"pre_vars": [("Pre", "scalar"),
                      ("PreRW", "scalar", VarAccess.READ_WRITE),
                      ("PreRO", "int", VarAccess.READ_ONLY)],
         "post_vars": [("Post", "scalar"),
                       ("PostRW", "scalar", VarAccess.READ_WRITE),
                       ("PostRO", "int", VarAccess.READ_ONLY)]},
        {}, {}, {"Pre": 1.0, "PreRW": 2.0, "PreRO": 3.0},
        {"Post": 4.0, "PostRW": 5.0, "PostRO": 6.0})

    reset_pre_vars = wum.reset_pre_vars
    assert len(reset_pre_vars) == 2
    assert reset_pre_vars[0] == ("Pre", "scalar", 1.0)
    assert reset_pre_vars[1] == ("PreRW", "scalar", 2.0)

    reset_post_vars = wum.reset_post_vars
    assert len(reset_post_vars) == 2
    assert reset_post_vars[0] == ("Post", "scalar", 4.0)
    assert reset_post_vars[1] == ("PostRW", "scalar", 5.0)