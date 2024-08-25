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
    """
    Enumeration of metric types for property calculations.

    Defines how a property is normalized or distributed across the system.

    Attributes:
        PER_STRUCTURE: Property is calculated for the entire structure.
        PER_ATOM: Property is normalized per atom in the structure.
        PER_SPECIES: Property is calculated separately for each atomic species.
    """
    PER_STRUCTURE = "per structure"
    PER_ATOM = "per atom"

@dataclass
class PropertyMetric:
    """
    Combines a physical property with a metric type for precise property representation.

    This class ensures that only valid combinations of properties and metrics are used,
    and provides methods to retrieve the appropriate units.

    Attributes:
        property_type (Property): The physical property being represented.
        metric_type (MetricType): The metric used for property calculation or normalization.
    """
    property_type: Property
    metric_type: MetricType

    def __post_init__(self):
        """
        Validates the combination of property type and metric type.

        Raises:
            ValueError: If the metric type is not applicable for the given property type.
        """
        valid_metrics = {
            Property.FORCES: [MetricType.PER_ATOM, MetricType.PER_STRUCTURE],
            Property.ENERGY: [MetricType.PER_ATOM, MetricType.PER_STRUCTURE],
            Property.STRESS: [MetricType.PER_ATOM, MetricType.PER_STRUCTURE]
        }
        if self.metric_type not in valid_metrics[self.property_type]:
            raise ValueError(f"Invalid metric type {self.metric_type} for property {self.property_type}")

    def get_units(self, add_metric: bool = True) -> str:
        """
          Retrieves the units for this property-metric combination.

          Args:
              add_metric (bool): If True, includes the metric in the unit string.

          Returns:
              str: The units, potentially including the metric (e.g., 'eV/atom' or just 'eV').

          Raises:
              ValueError: If an unknown metric type is encountered.
          """
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
