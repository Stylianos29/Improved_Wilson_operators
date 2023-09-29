import numpy as np
import inspect
import os

# __all__ = []

def compare_two_matrices(A, B) -> str:

  for precision in range(18):
    if not (np.isclose(A, B, atol=float("1e-"+str(precision))).all()):
      break
  return "Up to precision: 1e-"+str(precision-1)

# TODO: Construct a function that produces a better assessment of how close two values are

def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

def creating_directory_function(directory_path):
	if (not os.path.isdir(directory_path)):
		os.makedirs(directory_path)
		print(directory_path+" directory created.")
