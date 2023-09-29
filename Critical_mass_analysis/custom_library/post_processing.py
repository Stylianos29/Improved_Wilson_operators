'''Definitions of classes for the improved Wilson operators and overlap operators.'''
import numpy as np
import os
import sys

from custom_library.gauge import GaugeLinksField
import custom_library.constants as constants


# custom functions
def creating_directory_function(directory_path):
	if (not os.path.isdir(directory_path)):
		os.makedirs(directory_path)
		print(directory_path+" directory created.")

def jackknife_replicas_generation(array):
	'''INPUT: array of shape (N, lattice_size)'''
	N = np.shape(array)[0]

	jackknife_replicas = list()
	for row in range(N):
		truncated_array = np.delete(array, row, axis=0)
		jackknife_replicas.append(np.average(truncated_array, axis=0))
	return np.array(jackknife_replicas)

def jackknife_variance_array_function(jackknife_replicas_array):
	'''INPUT: jackknife replicas 2D array of shape (N, lattice_size)
	OUTPUT: array of shape (lattice_size,)'''
	jackknife_replicas_array = np.array(jackknife_replicas_array)
	Ν = np.shape(jackknife_replicas_array)[0]

	jackknife_average_array = np.average(jackknife_replicas_array, axis=0)
	shifted_jackknife_replicas_array = jackknife_replicas_array - jackknife_average_array
	jackknife_variance_array = ((Ν-1)/Ν)*np.sum(np.square(shifted_jackknife_replicas_array), axis=0)
	
	return jackknife_variance_array

def jackknife_covariance_matrix_function(jackknife_replicas_array):
	'''INPUT: jackknife replicas 2D array of shape (N, lattice_size)
	OUTPUT: covariance matrix 2D array of shape (lattice_size, lattice_size)
	'''
	jackknife_replicas_array = np.array(jackknife_replicas_array)
	Ν = np.shape(jackknife_replicas_array)[0]
	lattice_size = np.shape(jackknife_replicas_array)[1]

	# jackknife_average_array = np.average(jackknife_replicas_array, axis=0)
	# shifted_jackknife_replicas_array = jackknife_replicas_array - jackknife_average_array
	# temp = np.einsum("...i,...j->...ij", shifted_jackknife_replicas_array, shifted_jackknife_replicas_array)
	# jackknife_variance_array = ((Ν-1)/Ν)*np.sum(temp, axis=0)

	jackknife_variance_array = (Ν-1)*np.cov(jackknife_replicas_array, ddof=0, rowvar=False)
	
	return jackknife_variance_array

def effective_mass_periodic_case_function(array):

	lattice_size = np.shape(array)[0]
	middle_value_array = np.min(array) + (-1E-15)

	shifted_backward_array = np.roll(array, shift=+1)
	shifted_backward_array = shifted_backward_array[1:lattice_size//2-2]
	shifted_forward_array = np.roll(array, shift=-1)
	shifted_forward_array = shifted_forward_array[1:lattice_size//2-2]

	numerator = shifted_backward_array + np.sqrt(np.square(shifted_backward_array) - middle_value_array**2)
	denominator = shifted_forward_array + np.sqrt(np.square(shifted_forward_array) - middle_value_array**2)

	return 0.5*np.log(numerator/denominator)

def plateau_fit_function(x, p):
  return np.full(len(x), p)

def expfunc(x, a, b, lattice_size):
  return a *( np.exp(-b * x) + np.exp( -b*lattice_size ) * np.exp(  b * x ) )

def linear_func(x, p):
  '''Parameters: 1. p[0]: slope 2. p[1]: x-intercept (critical mass)'''
  x = np.array(x)
#   return p[0] + p[1]*x
  return p[0]*(x - p[1])

def quadratic_func(x, p):
  x = np.array(x)
#   return p[0] + p[1]*x + p[2]*np.square(x)
  return (p[0]*x - p[1])*(x - p[2])

def critical_kappa_value_function(critical_bare_mass_value):
	return 0.5/(critical_bare_mass_value + 4.0)