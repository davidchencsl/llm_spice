import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_spice.hardware.processor import Processor

from llm_spice.hardware.network import Network
from llm_spice.utils.breakdown import Breakdown
from llm_spice.utils.common import Si


HOURS_PER_YEAR = 24 * 365


@dataclass
class Server:
    chip: "Processor"
    network: Network
    num_chips: int = 8
    num_cpus: int = 2
    num_rams: int = 16
    num_ssds: int = 9

    cpu_cost: float = 3150.0
    ram_cost: float = 290.0
    ssd_cost: float = 168.0

    pcb_cost: float = 8775.0 - 50.0 * 11.75 - 390.0 * 8.0
    nic_cost: float = 1059.0
    mezz_board_cost: float = 2860.0
    vrm_current_rating: float = 70.0
    vrm_cost: float = 11.75
    psu_power_rating: float = 3000.0
    psu_cost: float = 390.0

    cooling_device_cost: float = 1850.0
    misc_cost: float = 1771.0 + 2256.34 + 6900.0 + 2996.0

    lifetime: float = 10.0

    c2c_bandwidth: float = 1.7 * Si.T
    c2c_latency: float = 6 * 1e-6
    n2n_bandwidth: float = 50 * Si.G

    def __post_init__(self):
        self.network.num_endpoints = self.num_chips

    @property
    def name(self) -> str:
        return f"Server ({self.num_chips}xChip)"

    def num_vrms(self) -> int:
        return math.ceil(
            self.chip.total_target_current() * self.num_chips / self.vrm_current_rating
        )

    def num_psus(self) -> int:
        return math.ceil(
            self.chip.total_power() * self.num_chips / self.psu_power_rating
        )

    def total_power(self) -> float:
        return self.chip.total_power() * self.num_chips

    def total_cost(self) -> float:
        return self.cost_breakdown().total()

    def power_breakdown(self) -> Breakdown:
        root = Breakdown.new_container(f"{self.name} Power")
        root.add(self.chip.power_breakdown().get_multiplied(self.num_chips))
        return root

    def misc_cost_breakdown(self) -> Breakdown:
        root = Breakdown.new_container(f"{self.name} Misc Cost")
        root.add(Breakdown.new_value("cpu_cost", self.cpu_cost * self.num_cpus))
        root.add(Breakdown.new_value("ram_cost", self.ram_cost * self.num_rams))
        root.add(Breakdown.new_value("ssd_cost", self.ssd_cost * self.num_ssds))
        root.add(Breakdown.new_value("nic_cost", self.nic_cost * self.num_chips))
        root.add(
            Breakdown.new_value(
                "mezz_board_cost", self.mezz_board_cost * self.num_chips
            )
        )
        root.add(Breakdown.new_value("vrm_cost", self.vrm_cost * self.num_vrms()))
        root.add(Breakdown.new_value("psu_cost", self.psu_cost * self.num_psus()))
        root.add(Breakdown.new_value("pcb_cost", self.pcb_cost))
        root.add(Breakdown.new_value("cooling_device_cost", self.cooling_device_cost))
        root.add(self.network.cost_breakdown())
        root.add(Breakdown.new_value("misc_cost", self.misc_cost))
        return root

    def cost_breakdown(self) -> Breakdown:
        root = Breakdown.new_container(f"{self.name} Cost")
        root.add(self.chip.cost_breakdown().get_multiplied(self.num_chips))
        root.add(self.misc_cost_breakdown())
        return root

    def capex_breakdown(self) -> Breakdown:
        root = Breakdown.new_container(f"{self.name} Capex")
        root.add(
            self.misc_cost_breakdown().get_multiplied(
                1.0 / (self.lifetime * HOURS_PER_YEAR)
            )
        )
        root.add(self.chip.capex_breakdown().get_multiplied(self.num_chips))
        return root


@dataclass
class Rack:
    server: Server
    num_servers: float = 4
    cooling_device_cost: float = 5500.0 + 24682.68 + 803.53
    # num_switches: int = 2
    # switch_cost: float = 27201.6

    misc_cost: float = 12076.0 + 2696.0

    switch_power: float = 10000.0
    misc_power: float = 16000.0  # With CPU power

    lifetime: float = 3.0

    @property
    def name(self) -> str:
        return f"Rack ({self.num_servers}xServer)"

    # Rack methods from components.rs
    def server_cost(self) -> float:
        return self.server.total_cost() * self.num_servers

    def total_power(self) -> float:
        return self.server.total_power() * self.num_servers

    def total_cost(self) -> float:
        return self.cost_breakdown().total()

    def power_breakdown(self) -> Breakdown:
        root = Breakdown.new_container(f"{self.name} Power")
        root.add(Breakdown.new_value("switch_power", self.switch_power))
        root.add(Breakdown.new_value("cooling_device_power", self.misc_power))
        root.add(self.server.power_breakdown().get_multiplied(self.num_servers))
        return root

    def misc_cost_breakdown(self) -> Breakdown:
        root = Breakdown.new_container(f"{self.name} Misc Cost")
        # root.add(
        #     Breakdown.new_value("switch_cost", self.switch_cost * self.num_switches)
        # )
        root.add(Breakdown.new_value("cooling_device_cost", self.cooling_device_cost))
        root.add(Breakdown.new_value("misc_cost", self.misc_cost))
        return root

    def cost_breakdown(self) -> Breakdown:
        root = Breakdown.new_container(f"{self.name} Cost")
        root.add(self.server.cost_breakdown().get_multiplied(self.num_servers))
        root.add(self.misc_cost_breakdown())
        return root

    def capex_breakdown(self) -> Breakdown:
        root = Breakdown.new_container(f"{self.name} Capex")
        root.add(
            self.misc_cost_breakdown().get_multiplied(
                1.0 / (self.lifetime * HOURS_PER_YEAR)
            )
        )
        root.add(self.server.capex_breakdown().get_multiplied(self.num_servers))
        return root


@dataclass
class Pod:
    rack: Rack
    network: Network
    num_racks: float = 1
    lifetime: float = 3.0

    def __post_init__(self):
        self.network.num_endpoints = math.ceil(self.total_num_chips())

    @property
    def name(self) -> str:
        return f"Pod ({self.num_racks}xRack)"

    def total_num_chips(self) -> float:
        return self.num_racks * self.rack.num_servers * self.rack.server.num_chips

    # Pod methods from components.rs
    def rack_cost(self) -> float:
        return self.rack.total_cost() * self.num_racks

    def total_cost(self) -> float:
        return self.cost_breakdown().total()

    def power_breakdown(self) -> Breakdown:
        root = Breakdown.new_container(f"{self.name} Power")
        root.add(self.rack.power_breakdown().get_multiplied(self.num_racks))
        return root

    def misc_cost_breakdown(self) -> Breakdown:
        root = Breakdown.new_container(f"{self.name} Misc Cost")
        root.add(self.network.cost_breakdown())
        return root

    def cost_breakdown(self) -> Breakdown:
        root = Breakdown.new_container(f"{self.name} Cost")
        root.add(self.rack.cost_breakdown().get_multiplied(self.num_racks))
        root.add(self.misc_cost_breakdown())
        return root

    def capex_breakdown(self) -> Breakdown:
        root = Breakdown.new_container(f"{self.name} Capex")
        root.add(
            self.misc_cost_breakdown().get_multiplied(
                1.0 / (self.lifetime * HOURS_PER_YEAR)
            )
        )
        root.add(self.rack.capex_breakdown().get_multiplied(self.num_racks))
        return root


@dataclass
class DataCenter:
    pod: Pod
    num_pods: float = 1

    infra_cost_per_watt: float = 5.0
    electricity_cost_per_kwh: float = 0.1
    cooling_cost_per_kwh: float = 0.03
    lifetime: float = 10.0

    @property
    def name(self) -> str:
        return f"DataCenter ({self.num_pods}xPod)"

    def pod_cost(self) -> float:
        return self.pod.total_cost() * self.num_pods

    def total_cost(self) -> float:
        return self.cost_breakdown().total()

    def total_power(self) -> float:
        return self.power_breakdown().total()

    def total_num_servers(self) -> float:
        return self.num_pods * self.pod.num_racks * self.pod.rack.num_servers

    def misc_cost_breakdown(self) -> Breakdown:
        root = Breakdown.new_container(f"{self.name} Misc Cost")
        root.add(
            Breakdown.new_value(
                "infra_cost", self.infra_cost_per_watt * self.total_power()
            )
        )
        return root

    def cost_breakdown(self) -> Breakdown:
        root = Breakdown.new_container(f"{self.name} Cost")
        root.add(self.pod.cost_breakdown().get_multiplied(self.num_pods))
        return root

    def power_breakdown(self) -> Breakdown:
        root = Breakdown.new_container(f"{self.name} Power")
        root.add(self.pod.power_breakdown().get_multiplied(self.num_pods))
        return root

    def capex_breakdown(self) -> Breakdown:
        root = Breakdown.new_container(f"{self.name} Capex")
        root.add(
            self.misc_cost_breakdown().get_multiplied(
                1.0 / (self.lifetime * HOURS_PER_YEAR)
            )
        )
        root.add(self.pod.capex_breakdown().get_multiplied(self.num_pods))
        return root

    def tco_breakdown(self) -> Breakdown:
        root = Breakdown.new_container(f"{self.name} TCO")
        root.add(self.capex_breakdown().rename("CapEx"))
        root.add(
            self.power_breakdown()
            .get_multiplied(
                (self.electricity_cost_per_kwh + self.cooling_cost_per_kwh) / 1000.0
            )
            .rename("OpEx")
        )
        return root
