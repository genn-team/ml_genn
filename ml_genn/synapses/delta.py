from .synapse import Model, Synapse

genn_model = {
    "apply_input_code":
        """
        $(Isyn) += $(inSyn);
        """,
    "decay_code":
        """
        $(inSyn) = 0;
        """}
        
class Delta(Synapse):
    def __init__(self):
        super(Delta, self).__init__()

    def get_model(self, population, dt):
        return Model(genn_model, {}, {})
