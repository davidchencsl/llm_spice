from llm_spice.hardware.network import Network
from llm_spice.op.operators import BaseOp, MoEFast
from llm_spice.utils.common import OpRunStats, DataType
from llm_spice.utils.registry import register_processor
from llm_spice.hardware.processor import Processor
from llm_spice.hardware.memory import Memory
from llm_spice.hardware.technode import TechNode


@register_processor(
    "GroqChip",
    hw_flops_bf16=188e12,
    hw_flops_fp8=188e12,
    hw_flops_fp4=188e12,
    hw_flops_efficiency=1.0,
    memory_bw_efficiency=1.0,
    memory=Memory.create("Groq-SRAM"),
    num_memory_banks=1,
    width=20,
    height=20,
    num_tiles=1,
    lifetime=10.0,
    tech_node=TechNode.create("PCB"),
    base_power=215,
    base_frequency=1.0,
    base_voltage=0.5,
    target_frequency=1.0,
    static_power_ratio=0.35,
    activity_factor=1.0,
    pod_size=1,
    rack_size=8,
    server_size=8,
    intranode_network=Network.create("NVLink-5-Switch"),
)
@register_processor(
    "WSE-3",
    hw_flops_bf16=125e15,
    hw_flops_fp8=125e15 * 2,
    hw_flops_fp4=125e15 * 4,
    hw_flops_efficiency=0.23,
    memory_bw_efficiency=0.5,
    memory=Memory.create("WSE-SRAM"),
    num_memory_banks=1,
    width=211,
    height=211,
    num_tiles=1,
    lifetime=10.0,
    tech_node=TechNode.create("WSI"),
    base_power=23e3,
    base_frequency=1.0,
    base_voltage=0.5,
    target_frequency=1.0,
    static_power_ratio=0.35,
    activity_factor=1.0,
    pod_size=1,
    rack_size=1,
    server_size=1,
)
@register_processor(
    "TPU-v5p",
    hw_flops_bf16=459e12,
    hw_flops_fp8=918e12,
    hw_flops_fp4=918e12,
    hw_flops_efficiency=0.8,
    memory_bw_efficiency=0.8,
    memory=Memory.create("HBM2E-8H"),
    num_memory_banks=6,
    width=33,
    height=24,
    num_tiles=1,
    lifetime=10.0,
    tech_node=TechNode.create("CoWoS"),
    base_power=959.0 - 6 * Memory.create("HBM2E-8H").power,
    base_frequency=2.04,
    base_voltage=0.5,
    target_frequency=1.0,
    static_power_ratio=0.35,
    activity_factor=1.0,
    pod_size=1,
    rack_size=4,
    server_size=8,
)
@register_processor(
    "TPU-v7p",
    hw_flops_bf16=4614e12 / 2,
    hw_flops_fp8=4614e12,
    hw_flops_fp4=4614e12,
    hw_flops_efficiency=0.8,
    memory_bw_efficiency=0.8,
    memory=Memory.create("HBM3E-8H"),
    num_memory_banks=8,
    width=21,
    height=21,
    num_tiles=2,
    lifetime=10.0,
    tech_node=TechNode.create("CoWoS"),
    base_power=959.0 - 8 * Memory.create("HBM3E-8H").power,
    base_frequency=1.633,
    base_voltage=0.5,
    target_frequency=1.0,
    static_power_ratio=0.35,
    activity_factor=1.0,
    pod_size=1,
    rack_size=4,
    server_size=8,
)
@register_processor(
    "H100",
    hw_flops_bf16=1979e12,
    hw_flops_fp8=1979e12 * 2,
    hw_flops_fp4=1979e12 * 4,
    hw_flops_efficiency=0.23,
    memory_bw_efficiency=0.5,
    memory=Memory.create("HBM3-8H"),
    num_memory_banks=5,
    width=33.0,
    height=26.0,
    num_tiles=1,
    lifetime=10.0,
    tech_node=TechNode.create("CoWoS"),
    base_power=700.0,
    base_frequency=1.83,
    base_voltage=0.9,
    target_frequency=1.83,
    static_power_ratio=0.35,
    activity_factor=1.0,
    pod_size=1,
    rack_size=4,
    server_size=8,
    intranode_network=Network.create("NVLink-4-Switch"),
)
@register_processor(
    "H200",
    hw_flops_bf16=1979e12,
    hw_flops_fp8=1979e12 * 2,
    hw_flops_fp4=1979e12 * 4,
    hw_flops_efficiency=0.4,
    memory_bw_efficiency=0.5,
    memory=Memory.create("HBM3E-8H"),
    num_memory_banks=6,
    width=33.0,
    height=26.0,
    num_tiles=1,
    lifetime=10.0,
    tech_node=TechNode.create("CoWoS"),
    base_power=700.0,
    base_frequency=1.83,
    base_voltage=0.9,
    target_frequency=1.83,
    static_power_ratio=0.35,
    activity_factor=1.0,
    pod_size=1,
    rack_size=4,
    server_size=8,
    intranode_network=Network.create("NVLink-4-Switch"),
)
@register_processor(
    "GB200",
    hw_flops_bf16=10e15 / 4,
    hw_flops_fp8=10e15 / 2,
    hw_flops_fp4=10e15,
    hw_flops_efficiency=0.3,
    memory_bw_efficiency=0.4,
    memory=Memory.create("HBM3E-12H"),
    num_memory_banks=8,
    width=31.0,
    height=25.8,
    num_tiles=2,
    lifetime=10.0,
    tech_node=TechNode.create("CoWoS"),
    base_power=1200.0,
    base_frequency=1.83,
    base_voltage=0.9,
    target_frequency=1.83,
    static_power_ratio=0.35,
    activity_factor=1.0,
    pod_size=1,
    rack_size=4,
    server_size=8,
    intranode_network=Network.create("NVLink-5-Switch"),
)
@register_processor(
    "GB300",
    hw_flops_bf16=15e15 / 4,
    hw_flops_fp8=15e15 / 2,
    hw_flops_fp4=15e15,
    hw_flops_efficiency=0.23,
    memory_bw_efficiency=0.5,
    memory=Memory.create("HBM3E-12H"),
    num_memory_banks=8,
    width=31.0,
    height=25.8,
    num_tiles=2,
    lifetime=10.0,
    tech_node=TechNode.create("CoWoS"),
    base_power=1400.0,
    base_frequency=1.83,
    base_voltage=0.9,
    target_frequency=1.83,
    static_power_ratio=0.35,
    activity_factor=1.0,
    pod_size=1,
    rack_size=4,
    server_size=8,
    intranode_network=Network.create("NVLink-5-Switch"),
)
@register_processor(
    "VR200",
    hw_flops_bf16=33.3e15 / 4,
    hw_flops_fp8=33.3e15 / 2,
    hw_flops_fp4=33.3e15,
    hw_flops_efficiency=0.23,
    memory_bw_efficiency=0.3,
    memory=Memory.create("HBM4-12H"),
    num_memory_banks=8,
    width=31.0,
    height=25.8,
    num_tiles=2,
    lifetime=10.0,
    tech_node=TechNode.create("CoWoS"),
    base_power=2300.0,
    base_frequency=1.83,
    base_voltage=0.9,
    target_frequency=1.83,
    static_power_ratio=0.35,
    activity_factor=1.0,
    pod_size=1,
    rack_size=4,
    server_size=8,
    intranode_network=Network.create("NVLink-5-Switch"),
)
@register_processor(
    "VR200-CPX",
    hw_flops_bf16=20e15 / 4,
    hw_flops_fp8=20e15 / 2,
    hw_flops_fp4=20e15,
    hw_flops_efficiency=0.23,
    memory_bw_efficiency=0.5,
    memory=Memory.create("GDDR7-16GB"),
    num_memory_banks=8,
    width=33.0,
    height=26.0,
    num_tiles=1,
    lifetime=10.0,
    tech_node=TechNode.create("PCB"),
    base_power=800,
    base_frequency=1.83,
    base_voltage=0.9,
    target_frequency=1.83,
    static_power_ratio=0.35,
    activity_factor=1.0,
    pod_size=1,
    rack_size=4,
    server_size=8,
    intranode_network=Network.create("NVLink-5-Switch"),
)
@register_processor(
    "6xHBM-PNM-1H",
    hw_flops_bf16=750e12,
    hw_flops_fp8=750e12 * 2,
    hw_flops_fp4=750e12 * 4,
    hw_flops_efficiency=0.5,
    memory_bw_efficiency=0.8,
    memory=Memory.create("HBM4-PNM-1H"),
    num_memory_banks=6,
    width=33.0,
    height=26.0,
    num_tiles=1,
    lifetime=10.0,
    tech_node=TechNode.create("CoWoS"),
    base_power=20 * 6,
    base_frequency=1.0,
    base_voltage=0.5,
    target_frequency=1.0,
    static_power_ratio=0.35,
    activity_factor=1.0,
    pod_size=1,
    rack_size=4,
    server_size=8,
    intranode_network=Network.create("NVLink-5-Switch"),
)
@register_processor(
    "6xHBM-PNM-2H",
    hw_flops_bf16=750e12,
    hw_flops_fp8=750e12 * 2,
    hw_flops_fp4=750e12 * 4,
    hw_flops_efficiency=0.5,
    memory_bw_efficiency=0.8,
    memory=Memory.create("HBM4-PNM-2H"),
    num_memory_banks=6,
    width=33.0,
    height=26.0,
    num_tiles=1,
    lifetime=10.0,
    tech_node=TechNode.create("CoWoS"),
    base_power=20 * 6,
    base_frequency=1.0,
    base_voltage=0.5,
    target_frequency=1.0,
    static_power_ratio=0.35,
    activity_factor=1.0,
    pod_size=1,
    rack_size=4,
    server_size=8,
    intranode_network=Network.create("NVLink-5-Switch"),
)
@register_processor(
    "6xHBM-PNM-4H",
    hw_flops_bf16=750e12,
    hw_flops_fp8=750e12 * 2,
    hw_flops_fp4=750e12 * 4,
    hw_flops_efficiency=0.5,
    memory_bw_efficiency=0.8,
    memory=Memory.create("HBM4-PNM-4H"),
    num_memory_banks=6,
    width=33.0,
    height=26.0,
    num_tiles=1,
    lifetime=10.0,
    tech_node=TechNode.create("CoWoS"),
    base_power=20 * 6,
    base_frequency=1.0,
    base_voltage=0.5,
    target_frequency=1.0,
    static_power_ratio=0.35,
    activity_factor=1.0,
    pod_size=1,
    rack_size=4,
    server_size=8,
    intranode_network=Network.create("NVLink-5-Switch"),
)
@register_processor(
    "6xHBM-PNM-8H",
    hw_flops_bf16=750e12,
    hw_flops_fp8=750e12 * 2,
    hw_flops_fp4=750e12 * 4,
    hw_flops_efficiency=0.5,
    memory_bw_efficiency=0.8,
    memory=Memory.create("HBM4-PNM-8H"),
    num_memory_banks=6,
    width=33.0,
    height=26.0,
    num_tiles=1,
    lifetime=10.0,
    tech_node=TechNode.create("CoWoS"),
    base_power=20 * 6,
    base_frequency=1.0,
    base_voltage=0.5,
    target_frequency=1.0,
    static_power_ratio=0.35,
    activity_factor=1.0,
    pod_size=1,
    rack_size=4,
    server_size=8,
    intranode_network=Network.create("NVLink-5-Switch"),
)
@register_processor(
    "Craftwerk",
    hw_flops_bf16=8e15,
    hw_flops_fp8=16e15,
    hw_flops_fp4=32e15,
    hw_flops_efficiency=0.5,
    memory_bw_efficiency=0.8,
    memory=Memory.create("HBM4-PNM-8H"),
    num_memory_banks=64,
    width=11,
    height=11,
    num_tiles=64,
    lifetime=10.0,
    tech_node=TechNode.create("CoWoS"),
    base_power=20 * 6,
    base_frequency=1.0,
    base_voltage=0.5,
    target_frequency=1.0,
    static_power_ratio=0.35,
    activity_factor=1.0,
    pod_size=1,
    rack_size=4,
    server_size=8,
    intranode_network=Network.create("NVLink-5-Switch"),
)
class SIMD(Processor):
    def __init__(
        self,
        hw_flops_bf16: float,
        hw_flops_fp8: float,
        hw_flops_fp4: float,
        hw_flops_efficiency: float = 1.0,
        memory_bw_efficiency: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hw_flops_bf16 = hw_flops_bf16
        self.hw_flops_fp8 = hw_flops_fp8
        self.hw_flops_fp4 = hw_flops_fp4
        self.hw_flops_efficiency = hw_flops_efficiency
        self.memory_bw_efficiency = memory_bw_efficiency

    def get_hw_flops(self, dtype: DataType) -> float:
        if dtype == DataType.BF16:
            return self.hw_flops_bf16
        elif dtype == DataType.FP8:
            return self.hw_flops_fp8
        elif dtype == DataType.FP4:
            return self.hw_flops_fp4
        else:
            raise ValueError(f"Unsupported dtype: {dtype}, {type(dtype)}")

    def get_effective_hw_flops(self, dtype: DataType) -> float:
        if dtype == DataType.BF16:
            flops = self.hw_flops_bf16
        elif dtype == DataType.FP8:
            flops = self.hw_flops_fp8
        elif dtype == DataType.FP4:
            flops = self.hw_flops_fp4
        else:
            raise ValueError(f"Unsupported dtype: {dtype}, {type(dtype)}")
        return flops * self.pcfg.tp_size * self.hw_flops_efficiency

    def get_effective_memory_bw(self) -> float:
        return self.memory_bw() * self.pcfg.tp_size * self.memory_bw_efficiency

    def execute_op(self, op: BaseOp) -> OpRunStats:
        """
        Run the roofline model analysis on an operator.

        The roofline model determines performance based on:
        1. Arithmetic intensity (FLOPS per byte of data movement)
        2. Peak FLOPS capability
        3. Peak memory bandwidth

        Returns OpRunStats with duration and utilization metrics.
        """
        assert op.is_leaf, "Should only evaluate leaf operators"

        OP_HANDLERS = {
            MoEFast: self._execute_moe,
        }

        for op_type in OP_HANDLERS:
            if isinstance(op, op_type):
                return OP_HANDLERS[op_type](op)

        return self._execute_default(op)

    def _execute_moe(self, op: MoEFast) -> OpRunStats:
        pcfg = self.pcfg.moe_pcfg(op.num_experts)
        total_flop = op.get_total_flop() / pcfg.ep_size
        total_memory_access_bytes = op.get_total_memory_access_bytes() / pcfg.ep_size

        compute_time = total_flop / (self.get_effective_hw_flops(op.dtype))
        memory_time = total_memory_access_bytes / (self.get_effective_memory_bw())

        duration = max(compute_time, memory_time)
        if duration == 0:
            return OpRunStats()

        flops_utilization = compute_time / duration
        memory_bw_utilization = memory_time / duration
        arithmetic_intensity = (
            total_flop / total_memory_access_bytes
            if total_memory_access_bytes > 0
            else 0
        )

        return OpRunStats(
            duration=duration,
            compute_time=compute_time,
            memory_time=memory_time,
            flop=total_flop,
            memory_access_bytes=total_memory_access_bytes,
            flops_utilization=flops_utilization,
            memory_bw_utilization=memory_bw_utilization,
            arithmetic_intensity=arithmetic_intensity,
        )

    def _execute_default(self, op: BaseOp) -> OpRunStats:
        total_flop = op.get_total_flop()
        total_memory_access_bytes = op.get_total_memory_access_bytes()
        compute_time = total_flop / self.get_effective_hw_flops(op.dtype)
        memory_time = total_memory_access_bytes / self.get_effective_memory_bw()

        duration = max(compute_time, memory_time)
        if duration == 0:
            return OpRunStats()

        flops_utilization = compute_time / duration
        memory_bw_utilization = memory_time / duration
        arithmetic_intensity = (
            total_flop / total_memory_access_bytes
            if total_memory_access_bytes > 0
            else 0
        )

        return OpRunStats(
            duration=duration,
            compute_time=compute_time,
            memory_time=memory_time,
            flop=total_flop,
            memory_access_bytes=total_memory_access_bytes,
            flops_utilization=flops_utilization,
            memory_bw_utilization=memory_bw_utilization,
            arithmetic_intensity=arithmetic_intensity,
        )
