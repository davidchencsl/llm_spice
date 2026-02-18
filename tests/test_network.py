from llm_spice.hardware.network import Network
from llm_spice.utils.common import format_value

def test_all_combinations():
    for network_name in Network.get_all_networks():
        network = Network.create(network_name)
        for collective in Network.Collectives:
            num_participants = 8
            data_size = 1024 * 1024
            print(f"{network_name} {collective} {num_participants} {format_value(data_size, 'iB')} {format_value(network.collective_time(collective, num_participants, data_size), 's')}")

def test_num_switches_and_links():
    th5 = Network.create("Tomahawk-5-256x200GbE")
    true_num_switches = [1, 1, 6, 12]
    true_num_links = [20, 256, 1024, 2048]
    for i, n in enumerate([20, 256, 512, 1024]):
        th5.num_endpoints = n
        print(f"switches: {n} {th5.num_switches}")
        print(f"links: {n} {th5.num_links}")
        assert th5.num_switches == true_num_switches[i]
        assert th5.num_links == true_num_links[i]

def test_cost():
    th5 = Network.create("Tomahawk-5-256x200GbE")
    print(th5.cost_breakdown())

