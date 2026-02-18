from llm_spice.hardware.memory import Memory, StackedDRAM
from llm_spice.utils.common import Si
import pytest

def test_stacked_dram():
    for h in [1, 2, 4, 8]:
        dram = Memory.create(f"HBM4-PNM-{h}H")
        assert isinstance(dram, StackedDRAM)
        print(dram.pretty_str())
    for h in [8, 12]:
        dram = Memory.create(f"HBM3E-{h}H")
        assert isinstance(dram, StackedDRAM)
        print(dram.pretty_str())
    dram = Memory.create("HBM3-8H")
    assert isinstance(dram, StackedDRAM)
    print(dram.pretty_str())
    dram = Memory.create("HBM4-12H")
    assert isinstance(dram, StackedDRAM)
    print(dram.pretty_str())

    
def test_hbm3e_8h():
    dram = Memory.create("HBM3E-8H")
    assert isinstance(dram, StackedDRAM)
    assert dram.capacity == 24 * Si.Gi
    assert dram.memory_bw == 1024 * Si.G
    assert dram.power == pytest.approx(38.1, rel=1e-3)
    assert dram.max_base_die_power == pytest.approx(39.31, rel=1e-3)
    