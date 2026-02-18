from llm_spice.allocator.allocator import MemoryUsage
from llm_spice import Allocator, Model, Processor, ParallelismConfig as ParCfg

def test_memory_usage():
    model = Model.create("llama3_405b")

    allocator = Allocator(model, 1000, 1000, 1024, 1000)
    
    vr200 = Processor.create("VR200", pcfg=ParCfg(tp_size=8))
    wse = Processor.create("WSE-3", pcfg=ParCfg(pp_size=64))

    memory_usage = allocator.memory_usage([vr200, wse])
    print(memory_usage)