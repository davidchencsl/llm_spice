from enum import Enum
import math


from llm_spice.utils.registry import register_network, NETWORK_REGISTRY
from llm_spice.utils.breakdown import Breakdown


class Collectives(Enum):
    ALL_GATHER = "all_gather"
    ALL_REDUCE = "all_reduce"
    A2A = "a2a"
    BROADCAST = "broadcast"
    REDUCE_SCATTER = "reduce_scatter"
    P2P = "p2p"


class Topology(Enum):
    RING = "ring"
    SWITCH = "switch"
    ALL_TO_ALL = "all_to_all"
    HYPERCUBE = "hypercube"
    BIPARTITE = "bipartite"  # 2D bipartite fully connected fabric: 2 * (n // 2), n = num_participants

    def all_gather_time(self, n: int, s: float, lat: float, bw: float):
        if n <= 1:
            return 0.0
        if bw <= 0:
            return float("inf")

        match self:
            case Network.Topology.SWITCH:
                return lat + (n - 1) * s / bw
            case Network.Topology.ALL_TO_ALL:
                return lat + s / bw
            case Network.Topology.HYPERCUBE:
                levels = math.ceil(math.log2(n))
                return levels * lat + (n - 1) * s / bw
            case Network.Topology.BIPARTITE:
                return 2 * lat + 2 * s / bw
            case Network.Topology.RING | _:  # default to ring
                return (n - 1) * lat + (n - 1) * s / bw

    def all_reduce_time(self, n: int, s: float, lat: float, bw: float):
        if n <= 1:
            return 0.0
        if bw <= 0:
            return float("inf")

        match self:
            case Network.Topology.SWITCH:
                return 2 * (lat + (n - 1) * s / bw)
            case Network.Topology.ALL_TO_ALL:
                levels = math.ceil(math.log2(n))
                return levels * (lat + s / bw)
            case Network.Topology.HYPERCUBE:
                levels = math.ceil(math.log2(n))
                return levels * (lat + s / bw)
            case Network.Topology.BIPARTITE:
                return 4 * lat + 2 * ((n - 1) / n) * s / bw
            case Network.Topology.RING | _:
                return 2 * ((n - 1) * lat + ((n - 1) / n) * s / bw)

    def broadcast_time(self, n: int, s: float, lat: float, bw: float):
        if n <= 1:
            return 0.0
        if bw <= 0:
            return float("inf")

        match self:
            case Network.Topology.SWITCH:
                return lat + (n - 1) * s / bw
            case Network.Topology.ALL_TO_ALL:
                return lat + s / bw
            case Network.Topology.HYPERCUBE:
                levels = math.ceil(math.log2(n))
                return levels * (lat + s / bw)
            case Network.Topology.BIPARTITE:
                return 2 * lat + 2 * s / bw
            case Network.Topology.RING | _:
                return (n - 1) * lat + (n - 1) * s / bw

    def reduce_scatter_time(self, n: int, s: float, lat: float, bw: float):
        if n <= 1:
            return 0.0
        if bw <= 0:
            return float("inf")

        match self:
            case Network.Topology.SWITCH:
                return lat + (n - 1) * s / bw
            case Network.Topology.ALL_TO_ALL:
                levels = math.ceil(math.log2(n))
                return levels * (lat + s / bw)
            case Network.Topology.HYPERCUBE:
                levels = math.ceil(math.log2(n))
                return levels * lat + ((n - 1) / n) * s / bw
            case Network.Topology.BIPARTITE:
                return 2 * lat + ((n - 1) / n) * s / bw
            case Network.Topology.RING | _:
                return (n - 1) * lat + ((n - 1) / n) * s / bw

    def a2a_time(self, n: int, s: float, lat: float, bw: float):
        if n <= 1:
            return 0.0
        if bw <= 0:
            return float("inf")

        match self:
            case Network.Topology.SWITCH:
                return (n - 1) * lat + s / bw
            case Network.Topology.ALL_TO_ALL:
                return lat + s / bw
            case Network.Topology.HYPERCUBE:
                levels = math.ceil(math.log2(n))
                return levels * lat + s / bw
            case Network.Topology.BIPARTITE:
                return 2 * lat + s / bw
            case Network.Topology.RING | _:
                return (n - 1) * lat + s / bw

    def p2p_time(self, n: int, s: float, lat: float, bw: float):
        if n <= 1:
            return 0.0
        if bw <= 0:
            return float("inf")

        match self:
            case Network.Topology.SWITCH:
                return lat + s / bw
            case Network.Topology.ALL_TO_ALL:
                return lat + s / bw
            case Network.Topology.HYPERCUBE:
                levels = math.ceil(math.log2(n))
                return levels * lat + s / bw
            case Network.Topology.BIPARTITE:
                return 2 * lat + s / bw
            case Network.Topology.RING | _:
                return lat + s / bw

    def collective_time(
        self,
        collective: Collectives,
        n: int,
        s: float,
        lat: float,
        bw: float,
    ):
        COLLECTIVES_FN = {
            Collectives.ALL_GATHER: self.all_gather_time,
            Collectives.ALL_REDUCE: self.all_reduce_time,
            Collectives.BROADCAST: self.broadcast_time,
            Collectives.REDUCE_SCATTER: self.reduce_scatter_time,
            Collectives.A2A: self.a2a_time,
            Collectives.P2P: self.p2p_time,
        }
        return COLLECTIVES_FN[collective](n, s, lat, bw)


@register_network(
    name="NVLink-4-Switch",
    topology=Topology.SWITCH,
    num_endpoints=8,
    endpoint_bw=450e9,  # 450GB/s bidirectional
    latency=5.0e-6,  # 1us
    switch_radix=8,
    cost_per_link=0.0,
    cost_per_switch=2000.0,
)
@register_network(
    name="NVLink-5-Switch",
    topology=Topology.SWITCH,
    num_endpoints=8,
    endpoint_bw=900e9,  # 900GB/s bidirectional
    latency=5.0e-6,  # 1us
    switch_radix=8,
    cost_per_link=0.0,
    cost_per_switch=2000.0,
)
@register_network(
    name="Tomahawk-5-64x800GbE",
    topology=Topology.SWITCH,
    num_endpoints=64,
    endpoint_bw=100e9,  # 800Gbps / 100GBps
    latency=1e-6,  # 1us
    switch_radix=64,
    cost_per_link=800.0,
    cost_per_switch=23e3,  # 23k USD
)
@register_network(
    name="Tomahawk-5-128x400GbE",
    topology=Topology.SWITCH,
    num_endpoints=128,
    endpoint_bw=50e9,
    latency=1e-6,  # 1us
    switch_radix=128,
    cost_per_link=400.0,
    cost_per_switch=23e3,  # 23k USD
)
@register_network(
    name="Tomahawk-5-256x200GbE",
    topology=Topology.SWITCH,
    num_endpoints=256,
    endpoint_bw=25e9,
    latency=1e-6,  # 1us
    switch_radix=256,
    cost_per_link=200.0,
    cost_per_switch=23e3,  # 23k USD
)
@register_network(
    name="IdealNetwork",
    topology=Topology.SWITCH,
    num_endpoints=32648,
    endpoint_bw=10e20,
    latency=0.0,
    switch_radix=32648,
    cost_per_link=0,
    cost_per_switch=0,
)
class Network:
    Topology = Topology
    Collectives = Collectives

    def __init__(
        self,
        name: str,
        topology: Topology,
        num_endpoints: int,
        endpoint_bw: float,
        latency: float,
        switch_radix: int,
        cost_per_link: float,
        cost_per_switch: float,
    ):
        self.name = name
        self.topology = topology
        self.num_endpoints = num_endpoints
        self.endpoint_bw = endpoint_bw
        self.latency = latency
        self.switch_radix = switch_radix
        self.cost_per_link = cost_per_link
        self.cost_per_switch = cost_per_switch

    @staticmethod
    def create(name: str, **kwargs) -> "Network":
        cls, kwargs = NETWORK_REGISTRY[name]
        kwargs.update(kwargs)
        return cls(**kwargs)

    @staticmethod
    def get_all_networks() -> list[str]:
        return list(NETWORK_REGISTRY.keys())

    @property
    def num_switches(self) -> int:
        if self.num_endpoints <= 1:
            return 0
        match self.topology:
            case Topology.SWITCH:
                # 2-level Clos (leaf-spine), non-blocking with symmetric radix.
                # Let k = switch_radix.
                # Each leaf uses d = floor(k/2) downlinks to endpoints and u = k - d uplinks to spines.
                # Number of leaf switches L = ceil(E / d).
                # Total uplink ports from leaves = L * u.
                # Each spine has k ports -> number of spines S = ceil((L * u) / k).
                # Total switches = L + S.
                k = self.switch_radix
                if k <= 1:
                    return 0
                if self.num_endpoints <= k:
                    return 1
                d = k // 2
                u = k - d
                if d == 0:
                    return 0
                L = math.ceil(self.num_endpoints / d)
                S = math.ceil((L * u) / k)
                return L + S
            case _:
                return 0

    @property
    def num_links(self) -> int:
        if self.num_endpoints <= 1:
            return 0
        match self.topology:
            case Topology.RING:
                # Simple ring among endpoints (one link between consecutive endpoints)
                return max(0, self.num_endpoints - 1)
            case Topology.SWITCH:
                # 2-level Clos (leaf-spine).
                # Host-to-leaf links: one per endpoint -> E.
                # Inter-switch links: total uplinks from leaves -> L * u.
                k = self.switch_radix
                if k <= 1:
                    return self.num_endpoints
                if self.num_endpoints <= k:
                    return self.num_endpoints
                d = k // 2
                u = k - d
                if d == 0:
                    return self.num_endpoints
                L = math.ceil(self.num_endpoints / d)
                inter_switch = L * u
                host_links = self.num_endpoints
                return host_links + inter_switch
            case Topology.ALL_TO_ALL:
                # Directed pairs; if counting bidirectional physical links, each pair is one link
                return self.num_endpoints * (self.num_endpoints - 1)
            case Topology.HYPERCUBE:
                # d-dimensional hypercube with n nodes (approx for non-power-of-2: d = ceil(log2 n))
                d = math.ceil(math.log2(self.num_endpoints))
                return int(self.num_endpoints * d // 2)
            case Topology.BIPARTITE:
                # Complete bipartite with two equal partitions of size n/2
                return self.num_endpoints * (self.num_endpoints // 2)

    def cost_breakdown(self) -> Breakdown:
        breakdown = Breakdown.new_container(f"{self.name} cost")
        breakdown.add(
            Breakdown.new_value("link_cost", self.cost_per_link).get_multiplied(
                self.num_links
            )
        )
        breakdown.add(
            Breakdown.new_value("switch_cost", self.cost_per_switch).get_multiplied(
                self.num_switches
            )
        )
        return breakdown

    @property
    def cost(self) -> float:
        return self.cost_breakdown().total()

    def collective_time(
        self,
        collective: Collectives,
        num_participants: int,
        data_size: int,  # per participant in bytes
    ):
        n = num_participants
        s = data_size
        lat = self.latency
        bw = self.endpoint_bw

        assert n <= self.num_endpoints

        return self.topology.collective_time(collective, n, s, lat, bw)
