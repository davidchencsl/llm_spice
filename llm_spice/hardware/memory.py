from dataclasses import dataclass


from llm_spice.utils.common import Si, format_value
from llm_spice.utils.registry import register_memory, MEMORY_REGISTRY


@register_memory(
    "GDDR7-16GB",
    _capacity=16 * Si.Gi,
    _power=10.0,
    _memory_bw=250 * Si.G,
    width=11.0,
    height=11.0,
    cost=53.3,
)
@register_memory(
    "WSE-SRAM",
    _capacity=44 * Si.Gi,
    _power=0,
    _memory_bw=21 * Si.P,
    width=0,
    height=0,
    cost=0,
)
@register_memory(
    "Groq-SRAM",
    _capacity=230 * Si.Mi,
    _power=0,
    _memory_bw=80 * Si.T,
    width=0,
    height=0,
    cost=0,
)
@register_memory(
    "SRAM",
    _capacity=96 * Si.Mi,
    _power=62.0,
    _memory_bw=20 * Si.T,
    width=0,
    height=0,
    cost=0,
)
@dataclass
class Memory:
    name: str
    _capacity: int
    _power: float
    _memory_bw: int
    width: float
    height: float
    cost: float

    @property
    def area(self):
        return self.width * self.height

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def power(self) -> float:
        return self._power

    @property
    def memory_bw(self) -> int:
        return self._memory_bw

    def pretty_str(self) -> str:
        lines = [
            f"{'=' * 10} {self.name} {'=' * 10}",
            f"Capacity: {format_value(self.capacity, 'iB')}",
            f"Memory BW: {format_value(self.memory_bw, 'iB/s')}",
            f"Power: {format_value(self.power, 'W')}",
            f"Cost: {format_value(self.cost, '$')}",
            f"Area: {format_value(self.area, 'mm^2')}",
        ]
        return "\n".join(lines)

    @staticmethod
    def create(name: str) -> "Memory":
        cls, kwargs = MEMORY_REGISTRY[name]
        return cls(**kwargs)

    @staticmethod
    def get_all_memory() -> list[str]:
        return list(MEMORY_REGISTRY.keys())


@register_memory(
    "HBM2E-8H",
    width=11.0,  # mm
    height=11.0,  # mm
    cost=100.0,
    stack_height=8,
    banks_per_layer=64,
    bank_height=32,  # number of mats
    bank_width=32,  # number of mats in a row
    mat_height=512,
    mat_width=512,
    # I/O
    pin_count=1024,
    data_rate=3.2e9,  # Hz
)
@register_memory(
    "HBM3-8H",
    width=11.0,  # mm
    height=11.0,  # mm
    cost=200.0,
    stack_height=8,
    banks_per_layer=64,
    bank_height=32,  # number of mats
    bank_width=32,  # number of mats in a row
    mat_height=512,
    mat_width=512,
    # I/O
    pin_count=1024,
    data_rate=5.2e9,  # Hz
)
@register_memory(
    "HBM3E-8H",
    width=11.0,  # mm
    height=11.0,  # mm
    cost=375.0,
    stack_height=8,
    banks_per_layer=64,
    bank_height=48,  # number of mats
    bank_width=32,  # number of mats in a row
    mat_height=512,
    mat_width=512,
    # I/O
    pin_count=1024,
    data_rate=8e9,  # Hz
)
@register_memory(
    "HBM3E-12H",
    width=11.0,  # mm
    height=11.0,  # mm
    cost=500.0,
    stack_height=12,
    banks_per_layer=64,
    bank_height=48,  # number of mats
    bank_width=32,  # number of mats in a row
    mat_height=512,
    mat_width=512,
    # I/O
    pin_count=1024,
    data_rate=8e9,  # Hz
)
@register_memory(
    "HBM4-12H",
    width=11.0,  # mm
    height=11.0,  # mm
    cost=600.0,
    stack_height=12,
    banks_per_layer=64,
    bank_height=24,  # number of mats
    bank_width=64,  # number of mats in a row
    mat_height=512,
    mat_width=512,
    # I/O
    pin_count=2048,
    data_rate=10e9,  # Hz
    # Power
    # activation_energy_per_cell=2e-14,  # J
    # on_die_io_energy=1.5e-12,  # J
    off_die_io_energy=0.3e-12,  # J
)
@register_memory(
    "HBM4-PNM-8H",
    width=0.0,  # Memory Stacked on top of base die
    height=0.0,  # Memory Stacked on top of base die
    cost=1000.0,
    stack_height=8,
    banks_per_layer=64 * 16,
    bank_height=2,  # number of mats
    bank_width=32,  # number of mats in a row
    mat_height=512,
    mat_width=512,
    # I/O
    pin_count=1024 * 16,
    data_rate=8e9,  # Hz
    # Power
    # activation_energy_per_cell=3e-14,  # J
    on_die_io_energy=0.7e-12,  # J
    off_die_io_energy=0,  # J
    # Thermal
    cold_plate_temperature=30,  # C
)
@register_memory(
    "HBM4-PNM-4H",
    width=0.0,  # Memory Stacked on top of base die
    height=0.0,  # Memory Stacked on top of base die
    cost=1000.0 / 8 * 4,
    stack_height=4,
    banks_per_layer=64 * 16,
    bank_height=2,  # number of mats
    bank_width=32,  # number of mats in a row
    mat_height=512,
    mat_width=512,
    # I/O
    pin_count=1024 * 16,
    data_rate=8e9,  # Hz
    # Power
    # activation_energy_per_cell=3e-14,  # J
    on_die_io_energy=0.7e-12,  # J
    off_die_io_energy=0,  # J
    # Thermal
    cold_plate_temperature=30,  # C
)
@register_memory(
    "HBM4-PNM-2H",
    width=0.0,  # Memory Stacked on top of base die
    height=0.0,  # Memory Stacked on top of base die
    cost=1000.0 / 8 * 2,
    stack_height=2,
    banks_per_layer=64 * 16,
    bank_height=2,  # number of mats
    bank_width=32,  # number of mats in a row
    mat_height=512,
    mat_width=512,
    # I/O
    pin_count=1024 * 16,
    data_rate=8e9,  # Hz
    # Power
    # activation_energy_per_cell=3e-14,  # J
    on_die_io_energy=0.7e-12,  # J
    off_die_io_energy=0,  # J
    # Thermal
    cold_plate_temperature=30,  # C
)
@register_memory(
    "HBM4-PNM-1H",
    width=0.0,  # Memory Stacked on top of base die
    height=0.0,  # Memory Stacked on top of base die
    cost=1000.0 / 8 * 1,
    stack_height=1,
    banks_per_layer=64 * 16,
    bank_height=2,  # number of mats
    bank_width=32,  # number of mats in a row
    mat_height=512,
    mat_width=512,
    # I/O
    pin_count=1024 * 16,
    data_rate=8e9,  # Hz
    # Power
    # activation_energy_per_cell=3e-14,  # J
    on_die_io_energy=0.7e-12,  # J
    off_die_io_energy=0,  # J
    # Thermal
    cold_plate_temperature=30,  # C
)
@dataclass
class StackedDRAM(Memory):
    name: str
    width: float = 11.0  # mm
    height: float = 11.0  # mm
    cost: float = 400.0
    stack_height: int = 8
    banks_per_layer: int = 64
    bank_height: int = 48  # number of mats
    bank_width: int = 32  # number of mats in a row
    mat_height: int = 512
    mat_width: int = 512
    bank_clk: float = 500e6  # Hz

    # I/O
    pin_count: int = 1024
    data_rate: float = 8e9  # Hz

    # Power
    activation_energy_per_cell: float = 4.4e-14  # J/bit
    on_die_io_energy: float = 2.24e-12  # J/bit
    off_die_io_energy: float = 0.3e-12  # J/bit

    # Thermal
    top_thermal_resistance: float = 0.14  # K/W
    layer_thermal_resistance: float = 0.06  # k/W
    base_junction_temperature: float = 85  # C
    cold_plate_temperature: float = 45  # C
    ambient_temperature: float = 25  # C

    # parent class fields required for registration
    _power: float = 0.0
    _memory_bw: int = 0
    _capacity: int = 0

    @property
    def capacity(self) -> int:
        return (
            self.stack_height
            * self.banks_per_layer
            * self.bank_height
            * self.bank_width
            * self.mat_height
            * self.mat_width
            // 8
        )

    @property
    def internal_bw(self) -> int:
        return int(
            self.stack_height
            * self.banks_per_layer
            * self.bank_width
            * self.bank_clk
            / 8
        )

    @property
    def io_bw(self) -> int:
        return int(self.pin_count * self.data_rate / 8)

    @property
    def memory_bw(self) -> int:
        return min(self.internal_bw, self.io_bw)

    @property
    def layer_memory_bw(self) -> int:
        return round(self.memory_bw / self.stack_height)

    @property
    def activation_energy(self) -> float:
        return self.activation_energy_per_cell * self.bank_height

    @property
    def layer_power(self) -> float:
        return (
            self.layer_memory_bw
            * 8
            * (self.activation_energy + self.on_die_io_energy + self.off_die_io_energy)
        )

    @property
    def power(self) -> float:
        return self.layer_power * self.stack_height

    # ---------------- Thermal Model (Top cold plate @ 45°C) ----------------
    def base_temperature(self, p_base: float = 0.0) -> float:
        """
        Includes ALL n inter-layer segments (Base-L1 ... L{n-1}-Ln) plus the top segment.
        """
        n = self.stack_height
        pL = self.layer_power
        Rt = self.top_thermal_resistance
        Rl = self.layer_thermal_resistance

        # ΔT_base = Rt*(P_b + n*P_L) + Rl*[ n*P_b + P_L*n(n+1)/2 ]
        deltaT = Rt * (p_base + n * pL) + Rl * (n * p_base + pL * n * (n + 1) / 2.0)
        return self.cold_plate_temperature + deltaT

    def layer_temperature(self, j: int, p_base: float = 0.0) -> float:
        """
        j = 1 (bottom) ... n (top). Top is coldest (touching sink through Rt).
        """
        n = self.stack_height
        if not (1 <= j <= n):
            raise ValueError("layer index j must be in [1..stack_height]")
        pL = self.layer_power
        Rt = self.top_thermal_resistance
        Rl = self.layer_thermal_resistance

        k = n - j  # number of inter-layer segments above layer j

        # T(j) = T_sink + Rt*(P_b + n*P_L) + Rl*[ k*P_b + P_L * k*(k+1)/2 ]
        return (
            self.cold_plate_temperature
            + Rt * (p_base + n * pL)
            + Rl * (k * p_base + pL * k * (k + 1) / 2.0)
        )

    @property
    def max_base_die_power(self) -> float:
        """
        P_b limit from T_base ≤ T_base,max with corrected path length:
          Rt*(P_b + n*P_L) + Rl*[ n*P_b + P_L*n(n+1)/2 ] ≤ H
        => P_b ≤ ( H - P_L*(Rt*n + Rl*n(n+1)/2) ) / ( Rt + n*Rl )
        """
        n = self.stack_height
        pL = self.layer_power
        Rt = self.top_thermal_resistance
        Rl = self.layer_thermal_resistance

        headroom = self.base_junction_temperature - self.cold_plate_temperature
        numerator = headroom - pL * (Rt * n + Rl * n * (n + 1) / 2.0)
        denom = Rt + n * Rl
        if denom <= 0:
            return 0.0
        return max(0.0, float(numerator / denom))

    def temperature_gradient(self, p_base: float = 0.0) -> dict:
        n = self.stack_height
        return {
            "base": self.base_temperature(p_base),
            **{
                f"layer_{j}": self.layer_temperature(j, p_base) for j in range(1, n + 1)
            },
            "cold_plate": self.cold_plate_temperature,
        }

    def temperature_gradient_diagram(self, p_base: float | None = None) -> str:
        """Generate a visual diagram showing temperature gradient across the stacked HBM structure."""
        if p_base is None:
            p_base = self.max_base_die_power

        temps = self.temperature_gradient(p_base)
        n = self.stack_height

        # Build visual stack from top (coldest) to bottom (hottest)
        rows = []

        # Cold plate at top (heat sink)
        rows.append(f"  COLD PLATE (Heat Sink)  {temps['cold_plate']:5.1f} °C")

        # Memory layers from top (n) to bottom (1)
        for j in range(n, 0, -1):
            power_str = format_value(self.layer_power, "W")
            rows.append(
                f"  DRAM Layer {j:2d}           {temps[f'layer_{j}']:5.1f} °C   {power_str:>8s}  "
            )

        # Base die at bottom
        power_str = format_value(p_base, "W")
        rows.append(
            f"  BASE DIE (Logic)        {temps['base']:5.1f} °C   {power_str:>8s}  "
        )

        # Find max width and create box
        max_width = max(len(row) for row in rows)
        rows = [row.ljust(max_width) for row in rows]

        # Build final output with borders
        lines = ["╔" + "═" * max_width + "╗"]
        for i, row in enumerate(rows):
            lines.append("║" + row + "║")
            if i < len(rows) - 1:
                lines.append("╠" + "═" * max_width + "╣")
        lines.append("╚" + "═" * max_width + "╝")

        return "\n".join(lines)

    def pretty_str(self) -> str:
        lines = [
            f"{super().pretty_str()}",
            f"Internal BW: {format_value(self.internal_bw, 'iB/s')}",
            f"IO BW: {format_value(self.io_bw, 'iB/s')}",
            f"Activation Energy: {format_value(self.activation_energy, 'J/b')}",
            f"Layer Power: {format_value(self.layer_power, 'W')}",
            f"Max Base Die Power: {format_value(self.max_base_die_power, 'W')}",
            self.temperature_gradient_diagram(p_base=self.max_base_die_power),
        ]
        return "\n".join(lines)
