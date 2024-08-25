from enum import Enum
from dataclasses import dataclass

class Property(Enum):
    """Enum for different types of properties with their corresponding units."""
    ENERGY = ('energy', 'eV')
    FORCES = ('forces', 'eV/Å')
    STRESS = ('stress', 'eV/Å³')

    def __init__(self, key: str, unit: str):
        self.key = key
        self.unit = unit

    def get_units(self):
        return self.unit

    def get_name(self):
        return self.key

class MetricType(Enum):
    """Enumeration of metric types for property calculations."""
    PER_STRUCTURE = "per structure"
    PER_ATOM = "per atom"

@dataclass
class PropertyMetric:
    """Combines a physical property with a metric type for precise property representation."""
    property_type: Property
    metric_type: MetricType

    def __post_init__(self):
        valid_metrics = {
            Property.FORCES: [MetricType.PER_ATOM, MetricType.PER_STRUCTURE],
            Property.ENERGY: [MetricType.PER_ATOM, MetricType.PER_STRUCTURE],
            Property.STRESS: [MetricType.PER_ATOM, MetricType.PER_STRUCTURE]
        }
        if self.metric_type not in valid_metrics[self.property_type]:
            raise ValueError(f"Invalid metric type {self.metric_type} for property {self.property_type}")

    def get_units(self, add_metric: bool = True) -> str:
        base_units = self.property_type.unit
        if not add_metric:
            return base_units

        if self.metric_type == MetricType.PER_ATOM:
            return f"{base_units}/atom"
        elif self.metric_type == MetricType.PER_STRUCTURE:
            return f"{base_units}/structure"
        else:
            raise ValueError(f"Unknown metric type: {self.metric_type}")

    @classmethod
    def from_string(cls, property_name: str, per_atom: bool) -> 'PropertyMetric':
        """
        Creates a PropertyMetric instance from a property name and a boolean indicating per atom metric.

        Args:
            property_name (str): The name of the property (e.g., 'energy', 'forces', 'stress').
            per_atom (bool): If True, the metric type is PER_ATOM; otherwise, PER_STRUCTURE.

        Returns:
            PropertyMetric: The created PropertyMetric instance.

        Raises:
            ValueError: If the property name is invalid.
        """
        # Find the Property enum by matching the key
        property_type = next((prop for prop in Property if prop.key == property_name.lower()), None)

        if property_type is None:
            raise ValueError(f"Invalid property name: {property_name}")

        # Determine the metric type based on the boolean
        metric_type = MetricType.PER_ATOM if per_atom else MetricType.PER_STRUCTURE

        # Create and return the PropertyMetric instance
        return cls(property_type=property_type, metric_type=metric_type)
