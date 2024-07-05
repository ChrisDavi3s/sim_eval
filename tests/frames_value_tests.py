import unittest
import numpy as np
from unittest.mock import Mock, patch
from sim_eval import Frames, Property, PropertyCalculator

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

        self.vasp_calc = MockCalculator('vasp', self.vasp_energies, self.vasp_forces, self.vasp_stresses)
        self.nequip_calc = MockCalculator('nequip', self.nequip_energies, self.nequip_forces, self.nequip_stresses)

        self.vasp_calc.compute_properties(self.frames)
        self.nequip_calc.compute_properties(self.frames)

    def test_get_property_magnitude(self):
        # Test energy (magnitude is the actual value)
        # For energy, the magnitude is simply the energy value itself (can be negative)
        np.testing.assert_allclose(
            self.frames.get_property_magnitude(Property.ENERGY, self.vasp_calc),
            self.vasp_energies
        )

        # Test forces (magnitude is Euclidean norm for each atom)
        # For forces, the magnitude is the Euclidean norm of the force vector for each atom
        expected_force_magnitudes = np.linalg.norm(self.vasp_forces, axis=2)
        np.testing.assert_allclose(
            self.frames.get_property_magnitude(Property.FORCES, self.vasp_calc),
            expected_force_magnitudes
        )

        # Test stress (magnitude is von Mises stress)
        # For stress, the magnitude is the von Mises stress calculated from the stress tensor
        expected_stress_magnitudes = Frames.calculate_von_mises_stress(self.vasp_stresses)
        np.testing.assert_allclose(
            self.frames.get_property_magnitude(Property.STRESS, self.vasp_calc),
            expected_stress_magnitudes
        )

    def test_get_mae(self):
        # Test energy MAE
        # Energy MAE: average absolute energy difference per structure, averaged over all frames
        expected_energy_mae = np.mean(np.abs(self.vasp_energies - self.nequip_energies))
        np.testing.assert_allclose(
            self.frames.get_mae(Property.ENERGY, self.vasp_calc, self.nequip_calc),
            expected_energy_mae
        )

        # Test forces MAE
        # Forces MAE: average absolute difference in force magnitude per atom, averaged over all frames
        # Returns an array with one value per atom
        vasp_forces_mag = np.linalg.norm(self.vasp_forces, axis=2)
        nequip_forces_mag = np.linalg.norm(self.nequip_forces, axis=2)
        expected_forces_mae = np.mean(np.abs(vasp_forces_mag - nequip_forces_mag), axis=0)
        np.testing.assert_allclose(
            self.frames.get_mae(Property.FORCES, self.vasp_calc, self.nequip_calc),
            expected_forces_mae
        )

        # Test stress MAE
        # Stress MAE: average absolute difference in von Mises stress, averaged over all frames
        vasp_stress_mag = Frames.calculate_von_mises_stress(self.vasp_stresses)
        nequip_stress_mag = Frames.calculate_von_mises_stress(self.nequip_stresses)
        expected_stress_mae = np.mean(np.abs(vasp_stress_mag - nequip_stress_mag))
        np.testing.assert_allclose(
            self.frames.get_mae(Property.STRESS, self.vasp_calc, self.nequip_calc),
            expected_stress_mae
        )

    def test_get_rmse(self):
        # Test energy RMSE
        # Energy RMSE: root mean square energy difference per structure, averaged over all frames
        expected_energy_rmse = np.sqrt(np.mean((self.vasp_energies - self.nequip_energies)**2))
        np.testing.assert_allclose(
            self.frames.get_rmse(Property.ENERGY, self.vasp_calc, self.nequip_calc),
            expected_energy_rmse
        )

        # Test forces RMSE
        # Forces RMSE: root mean square difference in force magnitude per atom, averaged over all frames
        # Returns an array with one value per atom
        vasp_forces_mag = np.linalg.norm(self.vasp_forces, axis=2)
        nequip_forces_mag = np.linalg.norm(self.nequip_forces, axis=2)
        expected_forces_rmse = np.sqrt(np.mean((vasp_forces_mag - nequip_forces_mag)**2, axis=0))
        np.testing.assert_allclose(
            self.frames.get_rmse(Property.FORCES, self.vasp_calc, self.nequip_calc),
            expected_forces_rmse
        )

        # Test stress RMSE
        # Stress RMSE: root mean square difference in von Mises stress, averaged over all frames
        vasp_stress_mag = Frames.calculate_von_mises_stress(self.vasp_stresses)
        nequip_stress_mag = Frames.calculate_von_mises_stress(self.nequip_stresses)
        expected_stress_rmse = np.sqrt(np.mean((vasp_stress_mag - nequip_stress_mag)**2))
        np.testing.assert_allclose(
            self.frames.get_rmse(Property.STRESS, self.vasp_calc, self.nequip_calc),
            expected_stress_rmse
        )

    def test_get_correlation(self):
        # Test energy correlation
        # Energy correlation: Pearson correlation coefficient between reference and target energies across all frames
        expected_energy_corr = np.corrcoef(self.vasp_energies, self.nequip_energies)[0, 1]
        np.testing.assert_allclose(
            self.frames.get_correlation(Property.ENERGY, self.vasp_calc, self.nequip_calc),
            expected_energy_corr
        )

        # Test forces correlation
        # Forces correlation: Pearson correlation coefficient between reference and target force magnitudes
        # Calculated separately for each atom, averaged over all frames
        # Returns an array with one correlation value per atom
        vasp_forces_mag = np.linalg.norm(self.vasp_forces, axis=2)
        nequip_forces_mag = np.linalg.norm(self.nequip_forces, axis=2)
        expected_forces_corr = np.array([
            np.corrcoef(vasp_forces_mag[:, i], nequip_forces_mag[:, i])[0, 1]
            for i in range(self.num_atoms)
        ])
        np.testing.assert_allclose(
            self.frames.get_correlation(Property.FORCES, self.vasp_calc, self.nequip_calc),
            expected_forces_corr
        )

        # Test stress correlation
        # Stress correlation: Pearson correlation coefficient between reference and target von Mises stresses across all frames
        vasp_stress_mag = Frames.calculate_von_mises_stress(self.vasp_stresses)
        nequip_stress_mag = Frames.calculate_von_mises_stress(self.nequip_stresses)
        expected_stress_corr = np.corrcoef(vasp_stress_mag, nequip_stress_mag)[0, 1]
        np.testing.assert_allclose(
            self.frames.get_correlation(Property.STRESS, self.vasp_calc, self.nequip_calc),
            expected_stress_corr
        )

    def test_calculate_von_mises_stress(self):
        # Test the static method calculate_von_mises_stress
        # von Mises stress is a scalar value representing the overall stress state
        stress_tensor = np.array([1.0, 2.0, 3.0, 0.5, 0.5, 0.5])
        expected_von_mises = np.sqrt(0.5 * ((1.0 - 2.0)**2 + (2.0 - 3.0)**2 + (3.0 - 1.0)**2 + 6*(0.5**2 + 0.5**2 + 0.5**2)))
        np.testing.assert_allclose(
            Frames.calculate_von_mises_stress(stress_tensor),
            expected_von_mises
        )

if __name__ == '__main__':
    unittest.main()