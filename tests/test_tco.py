from llm_spice.hardware import Processor, Memory, TechNode
from llm_spice.utils.common import format_value
from llm_spice import Allocator

def test_memory():
    for memory in Memory.get_all_memory():
        print(memory)
        print(Memory.create(memory))

def test_all_chips_tco():
    for processor in Processor.get_all_processors():
        processor = Processor.create(processor)
        for k, v in processor.device_info().items():
            if type(v) == float or type(v) == int:
                print(f"{k}: {format_value(v)}")
            else:
                print(f"{k}: {v}")

def test_h100_tco():
    chip = Processor.create("H100")

    tco = chip.get_tco_breakdown(8 * 4)
    print(tco)
    print(chip.dc.cost_breakdown())

    tco = chip.get_tco_breakdown(8 * 22)
    print(tco)
    print(chip.dc.cost_breakdown())
    
    