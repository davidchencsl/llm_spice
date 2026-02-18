import math
import logging
from dataclasses import dataclass, field

from llm_spice.hardware.network import Network
from llm_spice.op.operators import BaseOp
from llm_spice.utils.common import OpRunStats, ParallelismConfig as ParCfg
from llm_spice.utils.registry import PROCESSOR_REGISTRY
from llm_spice import DataType
from llm_spice.utils.breakdown import Breakdown
from llm_spice.hardware.memory import Memory
from llm_spice.hardware.technode import TechNode

from llm_spice.hardware.datacenter import DataCenter, Pod, Rack, Server


WAFER_DIAMETER = 300.0
HOURS_PER_YEAR = 365.0 * 24.0


@dataclass
class Processor:
    name: str
    memory: Memory
    num_memory_banks: int
    width: float
    height: float
    num_tiles: int
    lifetime: float
    tech_node: TechNode

    base_power: float
    base_frequency: float
    base_voltage: float
    target_frequency: float
    static_power_ratio: float
    activity_factor: float

    pod_size: int
    rack_size: int
    server_size: int

    intranode_network: Network = field(
        default_factory=lambda: Network.create("NVLink-5-Switch")
    )
    internode_network: Network = field(
        default_factory=lambda: Network.create("Tomahawk-5-64x800GbE")
    )

    pcfg: ParCfg = field(default_factory=ParCfg)

    def __post_init__(self):
        self.set_num_chips(self.pod_size * self.rack_size * self.server_size)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def execute_op(self, op: BaseOp) -> OpRunStats:
        raise NotImplementedError

    def get_hw_flops(self, dtype: DataType) -> float:
        raise NotImplementedError

    def get_effective_memory_capacity(self) -> float:
        return self.memory_capacity() * self.pcfg.num_dp_shards

    def set_parallelism(self, pcfg: ParCfg):
        self.pcfg = pcfg

    def device_info(self) -> dict:
        return {
            "name": self.name,
            "memory_capacity": self.memory_capacity(),
            "memory_bw": self.memory_bw(),
            "power": self.power_breakdown(),
            "cost": self.cost_breakdown(),
            "pcfg": self.pcfg,
        }

    @staticmethod
    def create(name: str, **other_kwargs) -> "Processor":
        processor_cls, kwargs = PROCESSOR_REGISTRY[name]
        kwargs.update(other_kwargs)
        return processor_cls(**kwargs)

    @staticmethod
    def get_all_processors() -> list[str]:
        return list(PROCESSOR_REGISTRY.keys())

    def memory_capacity(self) -> float:
        return self.memory.capacity * self.num_memory_banks

    def memory_bw(self) -> float:
        return self.memory.memory_bw * self.num_memory_banks

    def lump_factor(self) -> float:
        return (
            1.0
            / self.base_frequency
            * (self.base_voltage - self.tech_node.threshold_voltage)
            ** self.tech_node.velocity_saturation_index
            / self.base_voltage
        )

    def base_static_power(self) -> float:
        return self.static_power_ratio * self.base_power

    def leakage_current(self) -> float:
        return self.base_static_power() / self.base_voltage

    def base_dynamic_power(self) -> float:
        return self.base_power - self.base_static_power()

    def capacitance(self) -> float:
        return (
            self.base_dynamic_power()
            * 2.0
            / self.base_frequency
            / (self.base_voltage**2.0)
            / self.activity_factor
        )

    def target_voltage(self) -> float:
        # This is a simplified version - in Rust it uses a solver
        # For now, return the base voltage scaled by frequency ratio
        freq_ratio = self.target_frequency / self.base_frequency
        if freq_ratio <= 1.0:
            return self.base_voltage
        else:
            # Approximate voltage scaling
            return self.base_voltage * (freq_ratio**0.5)

    def target_static_power(self) -> float:
        return self.leakage_current() * self.target_voltage()

    def target_dynamic_power(self) -> float:
        return (
            0.5
            * self.capacitance()
            * (self.target_voltage() ** 2.0)
            * self.target_frequency
            * self.activity_factor
        )

    def total_target_power(self) -> float:
        return self.target_static_power() + self.target_dynamic_power()

    def total_target_current(self) -> float:
        return self.total_target_power() / self.target_voltage()

    def die_power_breakdown(self) -> Breakdown:
        root = Breakdown.new_container(f"{self.name} Power")
        root.add(Breakdown.new_value("static_power", self.target_static_power()))
        root.add(Breakdown.new_value("dynamic_power", self.target_dynamic_power()))
        return root

    # Chip methods from components.rs
    def tile_area(self) -> float:
        return self.width * self.height

    def die_area(self) -> float:
        return self.tile_area() * self.num_tiles

    def total_area(self) -> float:
        return self.die_area() + self.memory.area * self.num_memory_banks

    def chip_yield_rate(self) -> float:
        return self.tech_node.chip_yield_rate(self.tile_area())

    def interposer_yield_rate(self) -> float:
        return self.tech_node.interposer_yield_rate(self.total_area())

    def tile_cost(self) -> float:
        num_chips = self.tech_node.num_chips_per_wafer(
            self.width, self.height, WAFER_DIAMETER
        )
        yield_rate = self.chip_yield_rate()
        return self.tech_node.chip_wafer_cost / (yield_rate * num_chips)

    def memory_cost(self) -> float:
        return self.num_memory_banks * self.memory.cost

    def interposer_cost(self) -> float:
        height_num_tiles = math.ceil(math.sqrt(self.num_tiles))
        width_num_tiles = self.num_tiles // height_num_tiles
        height = self.height * height_num_tiles + self.memory.height * 2
        width = self.width * width_num_tiles

        if width * height < self.total_area():
            logging.info(
                f"Default memory physical layout not possible, using square layout: {width}x{height} <= {self.total_area()}"
            )
            width = math.sqrt(self.total_area())
            height = width

        num_interposers = self.tech_node.num_chips_per_wafer(
            width, height, WAFER_DIAMETER
        )
        assert num_interposers > 0, (
            f"width: {width}, height: {height}, num_interposers: {num_interposers}"
        )
        return (
            self.tech_node.interposer_wafer_cost
            / (self.interposer_yield_rate() * num_interposers)
            + self.tech_node.integration_cost
        )

    def integration_loss(self) -> float:
        return (
            1.0 - self.tech_node.integration_yield_rate
        ) * self.tech_node.integration_cost

    def total_cost(self) -> float:
        return self.cost_breakdown().total()

    def memory_power(self) -> float:
        return self.memory.power * self.num_memory_banks

    def total_power(self) -> float:
        return self.power_breakdown().total()

    def power_breakdown(self) -> Breakdown:
        root = Breakdown.new_container(f"{self.name} Power")
        root.add(Breakdown.new_value("memory_power", self.memory_power()))
        root.add(self.die_power_breakdown())
        return root

    def cost_breakdown(self) -> Breakdown:
        root = Breakdown.new_container(f"{self.name} Cost")
        root.add(
            Breakdown.new_value("tile_cost", self.tile_cost()).get_multiplied(
                self.num_tiles
            )
        )
        root.add(Breakdown.new_value("memory_cost", self.memory_cost()))
        root.add(Breakdown.new_value("interposer_cost", self.interposer_cost()))
        root.add(Breakdown.new_value("integration_loss", self.integration_loss()))
        return root

    def capex_breakdown(self) -> Breakdown:
        return self.cost_breakdown().get_multiplied(
            1.0 / (self.lifetime * HOURS_PER_YEAR)
        )

    def set_num_chips(self, num_chips: int):
        num_chips_per_server = min(num_chips, self.server_size)
        num_servers_per_rack = min(num_chips / num_chips_per_server, self.rack_size)
        num_racks_per_pod = min(
            num_chips / num_chips_per_server / num_servers_per_rack, self.pod_size
        )
        num_pods = (
            num_chips / num_chips_per_server / num_servers_per_rack / num_racks_per_pod
        )

        self.dc = DataCenter(
            num_pods=num_pods,
            pod=Pod(
                network=self.internode_network,
                num_racks=num_racks_per_pod,
                rack=Rack(
                    num_servers=num_servers_per_rack,
                    server=Server(
                        chip=self,
                        network=self.intranode_network,
                        num_chips=num_chips_per_server,
                    ),
                ),
            ),
        )

    def get_tco_breakdown(self, num_chips: int) -> Breakdown:
        self.set_num_chips(num_chips)
        return self.dc.tco_breakdown()

    def get_tco(self, num_chips: int) -> float:
        return self.get_tco_breakdown(num_chips).total()
