import unittest
import numpy as np
from unittest.mock import Mock, patch
from sim_eval import Frames, Property, PropertyMetric, MetricType, PropertyCalculator, calculate_von_mises_stress


class MockCalculator(PropertyCalculator):
    def __init__(self, name, energy, forces, stress):
        self.name = name
        self._energy = energy
        self._forces = forces
        self._stress = stress

    def compute_properties(self, frames):
        for i, frame in enumerate(frames.frames):
            frame.info[f'{self.name}_total_energy'] = self._energy[i]
            frame.arrays[f'{self.name}_forces'] = self._forces[i]
            frame.info[f'{self.name}_stress'] = self._stress[i]

def create_mock_frames(num_frames=3, num_atoms=2):
    mock_frames = []
    for _ in range(num_frames):
        mock_frame = Mock()
        mock_frame.info = {}
        mock_frame.arrays = {}
        mock_frames.append(mock_frame)
    return mock_frames


class TestFrames(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.num_frames = 3
        cls.num_atoms = 2

        # VASP data (reference)
        cls.vasp_energies = np.array([-1.0, -1.5, -2.0])
        cls.vasp_forces = np.array([
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            [[1.5, 0.5, 0.0], [-1.5, -0.5, 0.0]],
            [[2.0, 1.0, 0.5], [-2.0, -1.0, -0.5]]
        ])
        cls.vasp_stresses = np.array([
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [1.5, 1.5, 1.5, 0.1, 0.1, 0.1],
            [2.0, 2.0, 2.0, 0.2, 0.2, 0.2]
        ])

        # NEQuIP data (to compare against)
        cls.nequip_energies = np.array([-0.9, -1.4, -1.9])
        cls.nequip_forces = np.array([
            [[0.9, 0.1, 0.0], [-0.9, -0.1, 0.0]],
            [[1.4, 0.4, 0.1], [-1.4, -0.4, -0.1]],
            [[1.9, 0.9, 0.4], [-1.9, -0.9, -0.4]]
        ])
        cls.nequip_stresses = np.array([
            [0.9, 0.9, 0.9, 0.1, 0.1, 0.1],
            [1.4, 1.4, 1.4, 0.2, 0.2, 0.2],
            [1.9, 1.9, 1.9, 0.3, 0.3, 0.3]
        ])

    @patch('sim_eval.frames.read')
    def setUp(self, mock_read):
        self.mock_frames = create_mock_frames(self.num_frames, self.num_atoms)
        mock_read.return_value = self.mock_frames
        self.frames = Frames('dummy_path')

        self.vasp_calc = MockCalculator('vasp',
                                        self.vasp_energies,
                                        self.vasp_forces,
                                        self.vasp_stresses)
        self.nequip_calc = MockCalculator('nequip',
                                          self.nequip_energies,
                                          self.nequip_forces,
                                          self.nequip_stresses)

        self.vasp_calc.compute_properties(self.frames)
        self.nequip_calc.compute_properties(self.frames)

    def test_get_property_magnitude(self):
        # Test energy per structure
        energy_metric_structure = PropertyMetric(Property.ENERGY,
                                                 MetricType.PER_STRUCTURE)
        np.testing.assert_allclose(
            self.frames.get_property_magnitude(energy_metric_structure,
                                               self.vasp_calc),
            self.vasp_energies
        )

        # Test forces per atom
        forces_metric_atom = PropertyMetric(Property.FORCES,
                                            MetricType.PER_ATOM)
        expected_force_magnitudes = np.linalg.norm(self.vasp_forces, axis=2)
        np.testing.assert_allclose(
            self.frames.get_property_magnitude(forces_metric_atom,
                                               self.vasp_calc),
            expected_force_magnitudes
        )

        # Test stress per structure
        stress_metric_structure = PropertyMetric(Property.STRESS,
                                                 MetricType.PER_STRUCTURE)
        expected_stress_magnitudes = calculate_von_mises_stress(
                                                            self.vasp_stresses)
        np.testing.assert_allclose(
            self.frames.get_property_magnitude(stress_metric_structure,
                                               self.vasp_calc),
            expected_stress_magnitudes
        )

    def test_get_mae(self):
        # Test energy MAE per structure
        energy_metric_structure = PropertyMetric(Property.ENERGY,
                                                 MetricType.PER_STRUCTURE)
        expected_energy_mae = np.mean(
            np.abs(self.vasp_energies - self.nequip_energies))
        expected_energy_mae = np.array([expected_energy_mae])
        np.testing.assert_allclose(
            self.frames.get_mae(energy_metric_structure,
                                self.vasp_calc,
                                self.nequip_calc),
            expected_energy_mae
        )

        # Test forces MAE per atom
        forces_metric_atom = PropertyMetric(Property.FORCES,
                                            MetricType.PER_ATOM)
        vasp_forces_mag = np.linalg.norm(self.vasp_forces, axis=2)
        nequip_forces_mag = np.linalg.norm(self.nequip_forces, axis=2)
        expected_forces_mae = np.mean(
            np.abs(vasp_forces_mag - nequip_forces_mag), axis=0)
        expected_forces_mae = np.array([expected_forces_mae])

        np.testing.assert_allclose(
            self.frames.get_mae(forces_metric_atom,
                                self.vasp_calc,
                                self.nequip_calc),
            expected_forces_mae
        )

        # Test stress MAE per structure
        stress_metric_structure = PropertyMetric(Property.STRESS,
                                                 MetricType.PER_STRUCTURE)
        vasp_stress_mag = calculate_von_mises_stress(self.vasp_stresses)
        nequip_stress_mag = calculate_von_mises_stress(self.nequip_stresses)
        expected_stress_mae = np.mean(
            np.abs(vasp_stress_mag - nequip_stress_mag))
        np.testing.assert_allclose(
            self.frames.get_mae(stress_metric_structure,
                                self.vasp_calc,
                                self.nequip_calc),
            expected_stress_mae
        )

    def test_get_rmse(self):
        # Test energy RMSE per structure
        energy_metric_structure = PropertyMetric(Property.ENERGY,
                                                 MetricType.PER_STRUCTURE)
        expected_energy_rmse = np.sqrt(
            np.mean((self.vasp_energies - self.nequip_energies)**2))
        np.testing.assert_allclose(
            self.frames.get_rmse(
                energy_metric_structure, self.vasp_calc, self.nequip_calc),
            expected_energy_rmse
        )

        # Test forces RMSE per atom
        forces_metric_atom = PropertyMetric(Property.FORCES, 
                                            MetricType.PER_ATOM)
        vasp_forces_mag = np.linalg.norm(self.vasp_forces, axis=2)
        nequip_forces_mag = np.linalg.norm(self.nequip_forces, axis=2)
        expected_forces_rmse = np.sqrt(
            np.mean((vasp_forces_mag - nequip_forces_mag)**2, axis=0))
        expected_forces_rmse = np.array([expected_forces_rmse])
        np.testing.assert_allclose(
            self.frames.get_rmse(forces_metric_atom,
                                 self.vasp_calc,
                                 self.nequip_calc),
            expected_forces_rmse
        )

        # Test stress RMSE per structure
        stress_metric_structure = PropertyMetric(Property.STRESS,
                                                 MetricType.PER_STRUCTURE)
        vasp_stress_mag = calculate_von_mises_stress(self.vasp_stresses)
        nequip_stress_mag = calculate_von_mises_stress(self.nequip_stresses)
        expected_stress_rmse = np.sqrt(
            np.mean((vasp_stress_mag - nequip_stress_mag)**2))
        np.testing.assert_allclose(
            self.frames.get_rmse(stress_metric_structure,
                                 self.vasp_calc,
                                 self.nequip_calc),
            expected_stress_rmse
        )

    def test_get_correlation(self):
        # Test energy correlation per structure
        energy_metric_structure = PropertyMetric(Property.ENERGY,
                                                 MetricType.PER_STRUCTURE)
        expected_energy_corr = np.corrcoef(self.vasp_energies, 
                                           self.nequip_energies)[0, 1]
        np.testing.assert_allclose(
            self.frames.get_correlation(energy_metric_structure,
                                        self.vasp_calc,
                                        self.nequip_calc),
            expected_energy_corr
        )

        # Test forces correlation per atom
        forces_metric_atom = PropertyMetric(Property.FORCES,
                                            MetricType.PER_ATOM)
        vasp_forces_mag = np.linalg.norm(self.vasp_forces, axis=2)
        nequip_forces_mag = np.linalg.norm(self.nequip_forces, axis=2)
        expected_forces_corr = np.array([
            np.corrcoef(vasp_forces_mag[:, i], nequip_forces_mag[:, i])[0, 1]
            for i in range(self.num_atoms)
        ])
        expected_forces_corr = np.array([expected_forces_corr])
        np.testing.assert_allclose(
            self.frames.get_correlation(forces_metric_atom,
                                        self.vasp_calc,
                                        self.nequip_calc),
            expected_forces_corr
        )

        # Test stress correlation per structure
        stress_metric_structure = PropertyMetric(Property.STRESS,
                                                 MetricType.PER_STRUCTURE)
        vasp_stress_mag = calculate_von_mises_stress(self.vasp_stresses)
        nequip_stress_mag = calculate_von_mises_stress(self.nequip_stresses)
        expected_stress_corr = np.corrcoef(vasp_stress_mag, nequip_stress_mag)[0, 1]
        np.testing.assert_allclose(
            self.frames.get_correlation(stress_metric_structure,
                                        self.vasp_calc,
                                        self.nequip_calc),
            expected_stress_corr
        )

    def test_calculate_von_mises_stress(self):
        # Test the static method calculate_von_mises_stress
        stress_tensor = np.array([1.0, 2.0, 3.0, 0.5, 0.5, 0.5])
        expected_von_mises = np.sqrt(0.5 * ((1.0 - 2.0)**2 + (2.0 - 3.0)**2 + (3.0 - 1.0)**2 + 6*(0.5**2 + 0.5**2 + 0.5**2)))
        np.testing.assert_allclose(
            calculate_von_mises_stress(stress_tensor),
            expected_von_mises
        )


if __name__ == '__main__':
    unittest.main()
