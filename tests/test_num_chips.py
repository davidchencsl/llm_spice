from llm_spice.hardware.technode import TechNode

def test_num_chips():
    tech_node = TechNode.create("CoWoS")
    assert 4 == tech_node.num_chips_per_wafer(100, 100, wafer_diameter=300)