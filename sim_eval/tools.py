from typing import List, Union
import numpy as np

def normalize_frame_selection(total_frames: int, frame_number: Union[int, List[int], slice, None] = None) -> List[int]:
  """
  Convert frame number input into a list of integer frame numbers.

  Args:
      total_frames (int): Total number of frames available.
      frame_number (Union[int, List[int], slice, None]): Frame number(s) to normalize.
                                                         Default is None, which selects all frames.

  Returns:
      List[int]: A list of frame numbers.

  Raises:
      ValueError: If the frame number input is invalid or out of bounds.
  """
  if isinstance(frame_number, int):
      if frame_number < 0 or frame_number >= total_frames:
          raise ValueError(f"Frame number {frame_number} is out of range [0, {total_frames-1}]")
      return [frame_number]

  elif isinstance(frame_number, list):
      if not all(isinstance(i, int) for i in frame_number):
          raise ValueError("All elements in the list must be integers")
      if any(i < 0 or i >= total_frames for i in frame_number):
          raise ValueError(f"Frame numbers must be in range [0, {total_frames-1}]")
      return frame_number

  elif isinstance(frame_number, slice):
      start, stop, step = frame_number.indices(total_frames)
      return list(range(start, stop, step))

  elif frame_number is None:
      return list(range(total_frames))

  else:
      raise ValueError(f"Invalid frame number type: {type(frame_number)}")

def calculate_von_mises_stress(stress_tensor):
    """
    Calculate the von Mises stress from a stress tensor.

    Args:
        stress_tensor (np.ndarray): Stress tensor in Voigt notation [xx, yy, zz, yz, xz, xy].

    Returns:
        np.ndarray: von Mises stress, shape matches input except last dimension is reduced to 1.
    """
    xx, yy, zz, yz, xz, xy = stress_tensor.T
    von_mises = np.sqrt(0.5 * ((xx - yy)**2 + (yy - zz)**2 + (zz - xx)**2 + 6*(yz**2 + xz**2 + xy**2)))
    return  np.array(von_mises).reshape(-1)
