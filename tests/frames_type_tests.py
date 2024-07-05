import unittest
import numpy as np
from unittest.mock import Mock, patch
from sim_eval import Frames, Property, PropertyCalculator

class MockCalculator(PropertyCalculator):
    def __init__(self, name):
        self.name = name

def create_mock_frames(num_frames=10, num_atoms=5):
    mock_frames = []
    for _ in range(num_frames):
        mock_frame = Mock()
        mock_frame.info = {
            'vasp_total_energy': np.random.rand(),
            'nequip_total_energy': np.random.rand(),
            'vasp_stress': np.random.rand(6),
            'nequip_stress': np.random.rand(6)
        }
        mock_frame.arrays = {
            'vasp_forces': np.random.rand(num_atoms, 3),
            'nequip_forces': np.random.rand(num_atoms, 3)
        }
        mock_frames.append(mock_frame)
    return mock_frames

class TestFrames(unittest.TestCase):
    @patch('sim_eval.frames.read')
    def setUp(self, mock_read):
        self.mock_frames = create_mock_frames()
        mock_read.return_value = self.mock_frames
        self.frames = Frames('dummy_path')
        self.vasp_calc = MockCalculator('vasp')
        self.nequip_calc = MockCalculator('nequip')

    def test_get_property_magnitude(self):
        print('------Energy------')
        self.assertIsInstance(self.frames.get_property_magnitude(Property.ENERGY, self.vasp_calc, 1), np.ndarray)
        self.assertIsInstance(self.frames.get_property_magnitude(Property.ENERGY, self.vasp_calc, slice(0, 5)), np.ndarray)
        
        print('------Stress------')
        self.assertIsInstance(self.frames.get_property_magnitude(Property.STRESS, self.vasp_calc, 1), np.ndarray)
        self.assertIsInstance(self.frames.get_property_magnitude(Property.STRESS, self.vasp_calc, slice(0, 5)), np.ndarray)
        
        print('------Forces------')
        self.assertIsInstance(self.frames.get_property_magnitude(Property.FORCES, self.vasp_calc, 1), np.ndarray)
        self.assertIsInstance(self.frames.get_property_magnitude(Property.FORCES, self.vasp_calc, slice(0, 5)), np.ndarray)

    def test_get_mae(self):
        print('------Energy------')
        self.assertIsInstance(self.frames.get_mae(Property.ENERGY, self.vasp_calc, self.nequip_calc, 1), (float, np.ndarray))
        self.assertIsInstance(self.frames.get_mae(Property.ENERGY, self.vasp_calc, self.nequip_calc, slice(0, 5)), (float, np.ndarray))
        self.assertIsInstance(self.frames.get_mae(Property.ENERGY, self.vasp_calc, [self.nequip_calc, self.nequip_calc], 1), list)
        self.assertIsInstance(self.frames.get_mae(Property.ENERGY, self.vasp_calc, [self.nequip_calc, self.nequip_calc], slice(0, 5)), list)

        print('------Stress------')
        self.assertIsInstance(self.frames.get_mae(Property.STRESS, self.vasp_calc, self.nequip_calc, 1), (float, np.ndarray))
        self.assertIsInstance(self.frames.get_mae(Property.STRESS, self.vasp_calc, self.nequip_calc, slice(0, 5)), (float, np.ndarray))
        self.assertIsInstance(self.frames.get_mae(Property.STRESS, self.vasp_calc, [self.nequip_calc, self.nequip_calc], 1), list)
        self.assertIsInstance(self.frames.get_mae(Property.STRESS, self.vasp_calc, [self.nequip_calc, self.nequip_calc], slice(0, 5)), list)

        print('------Forces------')
        self.assertIsInstance(self.frames.get_mae(Property.FORCES, self.vasp_calc, self.nequip_calc, 1), np.ndarray)
        self.assertIsInstance(self.frames.get_mae(Property.FORCES, self.vasp_calc, self.nequip_calc, slice(0, 5)), np.ndarray)
        self.assertIsInstance(self.frames.get_mae(Property.FORCES, self.vasp_calc, [self.nequip_calc, self.nequip_calc], 1), list)
        self.assertIsInstance(self.frames.get_mae(Property.FORCES, self.vasp_calc, [self.nequip_calc, self.nequip_calc], slice(0, 5)), list)

    def test_get_rmse(self):
        print('------Energy------')
        self.assertIsInstance(self.frames.get_rmse(Property.ENERGY, self.vasp_calc, self.nequip_calc, 1), (float, np.ndarray))
        self.assertIsInstance(self.frames.get_rmse(Property.ENERGY, self.vasp_calc, self.nequip_calc, slice(0, 5)), (float, np.ndarray))
        self.assertIsInstance(self.frames.get_rmse(Property.ENERGY, self.vasp_calc, [self.nequip_calc, self.nequip_calc], 1), list)
        self.assertIsInstance(self.frames.get_rmse(Property.ENERGY, self.vasp_calc, [self.nequip_calc, self.nequip_calc], slice(0, 5)), list)

        print('------Stress------')
        self.assertIsInstance(self.frames.get_rmse(Property.STRESS, self.vasp_calc, self.nequip_calc, 1), (float, np.ndarray))
        self.assertIsInstance(self.frames.get_rmse(Property.STRESS, self.vasp_calc, self.nequip_calc, slice(0, 5)), (float, np.ndarray))
        self.assertIsInstance(self.frames.get_rmse(Property.STRESS, self.vasp_calc, [self.nequip_calc, self.nequip_calc], 1), list)
        self.assertIsInstance(self.frames.get_rmse(Property.STRESS, self.vasp_calc, [self.nequip_calc, self.nequip_calc], slice(0, 5)), list)

        print('------Forces------')
        self.assertIsInstance(self.frames.get_rmse(Property.FORCES, self.vasp_calc, self.nequip_calc, 1), np.ndarray)
        self.assertIsInstance(self.frames.get_rmse(Property.FORCES, self.vasp_calc, self.nequip_calc, slice(0, 5)), np.ndarray)
        self.assertIsInstance(self.frames.get_rmse(Property.FORCES, self.vasp_calc, [self.nequip_calc, self.nequip_calc], 1), list)
        self.assertIsInstance(self.frames.get_rmse(Property.FORCES, self.vasp_calc, [self.nequip_calc, self.nequip_calc], slice(0, 5)), list)

    def test_get_correlation(self):
        print('------Energy------')
        self.assertIsInstance(self.frames.get_correlation(Property.ENERGY, self.vasp_calc, self.nequip_calc, 1), (float, np.ndarray))
        self.assertIsInstance(self.frames.get_correlation(Property.ENERGY, self.vasp_calc, self.nequip_calc, slice(0, 5)), (float, np.ndarray))

        print('------Stress------')
        self.assertIsInstance(self.frames.get_correlation(Property.STRESS, self.vasp_calc, self.nequip_calc, 1), (float, np.ndarray))
        self.assertIsInstance(self.frames.get_correlation(Property.STRESS, self.vasp_calc, self.nequip_calc, slice(0, 5)), (float, np.ndarray))

        print('------Forces------')
        self.assertIsInstance(self.frames.get_correlation(Property.FORCES, self.vasp_calc, self.nequip_calc, 1), np.ndarray)
        self.assertIsInstance(self.frames.get_correlation(Property.FORCES, self.vasp_calc, self.nequip_calc, slice(0, 5)), np.ndarray)

if __name__ == '__main__':
    unittest.main()