from dataclasses import dataclass
import math

from llm_spice.utils.registry import register_technode, TECHNODE_REGISTRY
from llm_spice.utils.common import Si


@register_technode(
    "WSI",
    threshold_voltage=0.15,
    velocity_saturation_index=1.6,
    chip_wafer_defect_density=0.14 / 25,
    chip_wafer_cost=20 * Si.K,
    interposer_wafer_defect_density=0,
    interposer_wafer_cost=0 * Si.K,
    integration_yield_rate=1,
    integration_cost=0,
)
@register_technode(
    "CoWoS",
    threshold_voltage=0.15,
    velocity_saturation_index=1.6,
    chip_wafer_defect_density=0.14,
    chip_wafer_cost=20 * Si.K,
    interposer_wafer_defect_density=0.014,
    interposer_wafer_cost=5 * Si.K,
    integration_yield_rate=0.95,
    integration_cost=500.0,
)
@register_technode(
    "PCB",
    threshold_voltage=0.15,
    velocity_saturation_index=1.6,
    chip_wafer_defect_density=0.14,
    chip_wafer_cost=20 * Si.K,
    interposer_wafer_defect_density=0,
    interposer_wafer_cost=200.0,
    integration_yield_rate=0.999,
    integration_cost=50.0,
)
@dataclass
class TechNode:
    name: str
    threshold_voltage: float
    velocity_saturation_index: float
    chip_wafer_defect_density: float
    chip_wafer_cost: float
    interposer_wafer_defect_density: float
    interposer_wafer_cost: float
    integration_yield_rate: float
    integration_cost: float

    @staticmethod
    def create(name: str) -> "TechNode":
        cls, kwargs = TECHNODE_REGISTRY[name]
        return cls(**kwargs)

    @staticmethod
    def get_all_technodes() -> list[str]:
        return list(TECHNODE_REGISTRY.keys())

    def calc_chips_single_orientation(
        self, chip_width: float, chip_height: float, wafer_diameter: float
    ) -> int:
        wafer_radius = wafer_diameter / 2.0
        max_count = 0

        # Try different grid offsets to optimize placement (common in wafer calculators)
        offset_steps = 10  # Number of offset positions to try

        for x_offset_step in range(offset_steps):
            for y_offset_step in range(offset_steps):
                x_offset = (x_offset_step / offset_steps) * chip_width
                y_offset = (y_offset_step / offset_steps) * chip_height

                count = 0

                # Generate grid with more positions to account for offset
                # Start from negative positions to ensure we include center
                rows = int(math.ceil((wafer_diameter + chip_height) / chip_height))
                cols = int(math.ceil((wafer_diameter + chip_width) / chip_width))

                # Start from negative indices to ensure we cover center region
                for row in range(-(rows // 2), (rows // 2) + 1):
                    for col in range(-(cols // 2), (cols // 2) + 1):
                        # Calculate chip center coordinates with offset
                        chip_center_x = col * chip_width - x_offset
                        chip_center_y = row * chip_height - y_offset

                        # Check if all four corners of the chip are within the wafer
                        corners = [
                            (
                                chip_center_x - chip_width / 2.0,
                                chip_center_y - chip_height / 2.0,
                            ),
                            (
                                chip_center_x + chip_width / 2.0,
                                chip_center_y - chip_height / 2.0,
                            ),
                            (
                                chip_center_x - chip_width / 2.0,
                                chip_center_y + chip_height / 2.0,
                            ),
                            (
                                chip_center_x + chip_width / 2.0,
                                chip_center_y + chip_height / 2.0,
                            ),
                        ]

                        all_corners_inside = all(
                            x * x + y * y <= wafer_radius * wafer_radius
                            for x, y in corners
                        )

                        if all_corners_inside:
                            count += 1

                max_count = max(max_count, count)

        return max_count

    def num_chips_per_wafer(
        self, chip_width: float, chip_height: float, wafer_diameter: float = 300.0
    ) -> int:
        # Try both orientations and return the maximum
        orientation1 = self.calc_chips_single_orientation(
            chip_width, chip_height, wafer_diameter
        )
        orientation2 = self.calc_chips_single_orientation(
            chip_height, chip_width, wafer_diameter
        )
        return max(orientation1, orientation2)

    def chip_yield_rate(self, chip_area: float) -> float:
        defect_density_per_mm2 = self.chip_wafer_defect_density / 100.0
        yield_rate = math.exp(-defect_density_per_mm2 * chip_area)
        return yield_rate

    def interposer_yield_rate(self, interposer_area: float) -> float:
        defect_density_per_mm2 = self.interposer_wafer_defect_density / 100.0
        yield_rate = math.exp(-defect_density_per_mm2 * interposer_area)
        return yield_rate
