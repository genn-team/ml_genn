from .compiler import Compiler
from .compiled_model import CompiledModel

#CompiledInferenceModel = type("CompiledInferenceModel", (CompiledModel, InferenceMixin), 
#    dict(CompiledModel.__dict__))

class InferenceCompiler(Compiler):
    def __init__(self, output_populations, dt:float=1.0, batch_size:int=1, rng_seed:int=0,
                 kernel_profiling:bool=False, prefer_in_memory_connect=True,
                 **genn_kwargs):
        super(InferenceCompiler, self).__init__(dt, batch_size, rng_seed,
                                                kernel_profiling, 
                                                prefer_in_memory_connect,
                                                genn_kwargs)
        self.output_populations = output_populations
    
    