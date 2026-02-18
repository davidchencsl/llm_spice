from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Union


@dataclass
class Breakdown:
    name: str
    value: Union[float, Dict[str, "Breakdown"]]

    @classmethod
    def new_value(cls, name: str, v: float) -> "Breakdown":
        return cls(name=name, value=v)

    @classmethod
    def new_container(cls, name: str) -> "Breakdown":
        return cls(name=name, value=OrderedDict())

    def rename(self, name: str) -> "Breakdown":
        return Breakdown(name=name, value=self.value)

    def add(self, child: "Breakdown"):
        if isinstance(self.value, dict):
            if child.name in self.value:
                self.value[child.name].merge(child)
            else:
                self.value[child.name] = child
        else:
            raise ValueError("attempted to add a child to a leaf node")

    def get_multiplied(self, factor: float) -> "Breakdown":
        if isinstance(self.value, (int, float)):
            return Breakdown(name=self.name, value=self.value * factor)
        elif isinstance(self.value, dict):
            new_children = OrderedDict()
            for k, v in self.value.items():
                new_children[k] = v.get_multiplied(factor)
            return Breakdown(name=self.name, value=new_children)
        else:
            raise ValueError(f"Unknown value type: {type(self.value)}")

    def merge(self, other: "Breakdown"):
        if isinstance(self.value, (int, float)) and isinstance(
            other.value, (int, float)
        ):
            self.value += other.value
        elif isinstance(self.value, dict) and isinstance(other.value, dict):
            for k, v in other.value.items():
                if k in self.value:
                    self.value[k].merge(v)
                else:
                    self.value[k] = v
        else:
            raise ValueError(f"structure mismatch while merging `{self.name}`")

    def total(self) -> float:
        if isinstance(self.value, (int, float)):
            return float(self.value)
        elif isinstance(self.value, dict):
            return sum(b.total() for b in self.value.values())
        else:
            return 0.0

    @staticmethod
    def percentage_to_color(percentage: float) -> str:
        """Convert percentage to ANSI color code"""
        # Clamp percentage to 0-100 range
        p = max(0.0, min(100.0, percentage))

        if p <= 50.0:
            # Green to Yellow
            ratio = p / 50.0
            r = int(255 * ratio)
            g = 255
            b = 0
        else:
            # Yellow to Red
            ratio = (p - 50.0) / 50.0
            r = 255
            g = int(255 * (1.0 - ratio))
            b = 0

        return f"\033[38;2;{r};{g};{b}m"

    @staticmethod
    def blue_color() -> str:
        return "\033[94m"

    @staticmethod
    def white_color() -> str:
        return "\033[97m"

    @staticmethod
    def reset_color() -> str:
        return "\033[0m"

    def _fmt_inner(
        self, prefix: str = "", is_last: bool = True, total: float | None = None
    ) -> str:
        if total is None:
            total = self.total()

        current = self.total()
        percentage = 100.0 if total == 0.0 else (100.0 * current / total)

        # Print the current node with appropriate tree characters
        branch = "└─ " if is_last else "├─ "

        # Create gradient color from red to green based on percentage
        percentage_color = self.percentage_to_color(percentage)
        percentage_str = f"{percentage_color}{percentage:.2f}%{self.reset_color()}"

        result = f"{prefix}{branch}[{percentage_str}] {self.blue_color()}{current:.2f}{self.reset_color()} {self.white_color()}{self.name}{self.reset_color()}\n"

        if isinstance(self.value, dict):
            children_list = list(self.value.values())
            extension = "   " if is_last else "│  "
            new_prefix = f"{prefix}{extension}"

            for i, child in enumerate(children_list):
                is_last_child = i == len(children_list) - 1
                result += child._fmt_inner(new_prefix, is_last_child, current)

        return result

    def __str__(self) -> str:
        return self._fmt_inner().rstrip()

    def __repr__(self) -> str:
        return self.__str__()
