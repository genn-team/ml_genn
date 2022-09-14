from ml_genn.utils.model import (CustomUpdateModel, NeuronModel,
                                 SynapseModel, WeightUpdateModel)

from pygenn.genn_wrapper.Models import (VarAccess_READ_ONLY,
                                        VarAccess_READ_WRITE)

from pytest import raises

def test_process():
    pass

def test_reset_vars():
    nm = NeuronModel({"var_name_types": [("V", "scalar"),
                                         ("VRW", "scalar", VarAccess_READ_WRITE),
                                         ("VRO", "int", VarAccess_READ_ONLY)]},
                    {}, {"V": 1.0, "VRW": 2.0, "VRO": 3.0})
    reset_vars = nm.reset_vars
    assert len(reset_vars) == 2
    assert reset_vars[0] == ("V", "scalar", 1.0)
    assert reset_vars[1] == ("VRW", "scalar", 2.0)

def test_weight_update_pre_post_reset_vars():
    wum = WeightUpdateModel(
        {"pre_var_name_types": [("Pre", "scalar"),
                                ("PreRW", "scalar", VarAccess_READ_WRITE),
                                ("PreRO", "int", VarAccess_READ_ONLY)],
         "post_var_name_types": [("Post", "scalar"),
                                 ("PostRW", "scalar", VarAccess_READ_WRITE),
                                 ("PostRO", "int", VarAccess_READ_ONLY)]},
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