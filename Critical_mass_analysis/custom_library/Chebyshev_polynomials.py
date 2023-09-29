import numpy as np
from numpy.linalg import matrix_power
from math import factorial
from scipy.linalg import fractional_matrix_power
from scipy import linalg

# import custom_library.auxiliary

# __all__ = []

def scalar_Chebyshev_polynomials_function(x, n):
  '''TODO: Write description'''

  Tn_minus1 = 1
  if (n == 0):
    return Tn_minus1
  else:
    Tn = x
    for order in range(1, n):
      temp = Tn
      Tn = 2*x*Tn - Tn_minus1
      Tn_minus1 = temp

    return Tn

def _matrix_sign_function_array_function(array):
  '''TODO: Write description'''

  array = np.array(array)
  array_norm_squared = np.matmul(array, np.conjugate(array.T))
  inverse_array_norm = fractional_matrix_power(array_norm_squared, -0.5)
  sign_function_array = np.matmul(array, inverse_array_norm)

  return sign_function_array

def _recursive_matrix_Chebyshev_polynomials_function(X, n):

	if (n == 0):
		return np.identity(np.shape(X)[0], dtype=np.complex_)
	elif (n == 1):
		return X
	else:
		return 2.0*np.matmul(X, _recursive_matrix_Chebyshev_polynomials_function(X, n-1)) - _recursive_matrix_Chebyshev_polynomials_function(X, n-2)

def matrix_Chebyshev_polynomials_function(X, n):

  Tn_minus1 = np.identity(np.shape(X)[0], dtype=np.complex_)

  if (n == 0):
    return Tn_minus1
  else:
    Tn = X
    for order in range(1, n):
      temp = Tn
      Tn = 2*np.matmul(X, Tn) - Tn_minus1
      Tn_minus1 = temp

    return Tn

def _scalar_approximate_cosine_function(x, n=10):

    assert (n >= 0) and isinstance(n, int), 'n must be a positive integer greater than 1.'

    sign = +1
    x = x%(2.*np.pi)
    if (x > np.pi/2.):
      if (x < np.pi):
        x = np.pi - x
        sign = -1
      elif (x < 3.*np.pi/2.):
        x = x - np.pi
        sign = -1
      else:
        x = 2*np.pi - x
        sign = +1
          
    sum = 1.
    for k in range(1, n):
        sum += ((-1)**(k)/factorial(2*k))*x**(2*k)

    return sign*sum

def _matrix_approximate_cosine_function(X, n=10):
    
    assert np.shape(X)[0] == np.shape(X)[1], 'Matrix argument must be a square matrix.'
    # assert (X <= np.pi/2.).all() and (X >= np.pi/2.).all(), 'The absolute value of all elements of the matrix must be lower than π/2.'
    assert (n >= 0) and isinstance(n, int), 'n must be a positive integer greater than 1.'

    matrix_dimension = np.shape(X)[0]
    squared_matrix = matrix_power(X, 2)
    sum = np.zeros(np.shape(X))
    # , dtype=np.complex_)
    for k in range(n+1):
        sum += ((-1)**(k)/factorial(2*k))*matrix_power(squared_matrix, k)
    
    return sum

def _matrix_approximate_sine_function(X, n=10):
    
    assert np.shape(X)[0] == np.shape(X)[1], 'Matrix argument must be a square matrix.'
    assert (n >= 0) and isinstance(n, int), 'n must be a positive integer greater than 1.'

    matrix_dimension = np.shape(X)[0]
    sum = np.zeros((matrix_dimension, matrix_dimension))
    for k in range(n+1):
        sum += ((-1)**(k)/factorial(2*k+1))*matrix_power(X, 2*k+1)
    
    return sum

# def Chebyshev_polynomials_column_function(n, x, b):

#   Tn_minus1 = b

#   if (n == 0):
#     return Tn_minus1

#   else:
#     Tn = x
#     for order in range(1, n):
#       temp = Tn
#       Tn = 2*(x*Tn) - Tn_minus1
#       Tn_minus1 = temp

#     return Tn

# def approximate_matrix_sign_function(array, N=50):

# 	eg_val, eg_vect = linalg.eig(array)

# 	alpha = np.min(np.absolute(eg_val))
# 	beta = np.max(np.absolute(eg_val))

# 	X = (1/(beta**2 - alpha**2))*( 2*linalg.fractional_matrix_power(array, 2) - (beta**2 + alpha**2)*np.identity(np.shape(array)[0], dtype=np.complex_) )

# 	temp_sum = np.zeros( np.shape(array), dtype=np.complex_)
# 	for n in range(N):
# 		temp_sum += Chebyshev_polynomials_factors(n, N, alpha, beta)*matrix_Chebyshev_polynomials_function(X, n)

# 	return np.matmul(array, temp_sum)

# # GAUGED OPERATORS

def Chebyshev_polynomials_factors(n, N, alpha, beta):

  sum = 0.0
  for k in range(1, N+1):
    xk = np.cos((k - 0.5)*np.pi/float(N))
    sum += r(xk, alpha, beta)*Chebyshev_polynomials_function(xk, n)

  # return np.pi/float(N)*sum
  if (n==0):
    factor = 1.0/float(N)
  else:
    factor = 2.0/float(N)
  
  return factor*sum


def recursive_Chebyshev_polynomials_matrix_function(X, n):
	if (n == 0):
		return np.identity(np.shape(X)[0], dtype=np.complex_)
	elif (n == 1):
		return X
	else:
		return 2.0*np.matmul(X, recursive_Chebyshev_polynomials_matrix_function(X, n-1)) - recursive_Chebyshev_polynomials_matrix_function(X, n-2)


def Chebyshev_polynomials_matrix_function(X, n):

  Tn_minus1 = np.identity(np.shape(X)[0], dtype=np.complex_)

  if (n==0):
    return Tn_minus1
  else:
    Tn = X
    for order in range(1, n):
      temp = Tn
      Tn = 2*np.matmul(X, Tn) - Tn_minus1
      Tn_minus1 = temp

    return Tn

def Chebyshev_polynomials_column_function(n, x, b):

  Tn_minus1 = b

  if (n==0):
    return Tn_minus1

  else:
    Tn = x
    for order in range(1, n):
      temp = Tn
      Tn = 2*(x*Tn) - Tn_minus1
      Tn_minus1 = temp

    return Tn


def approximate_matrix_sign_function(array, N=50):

	# eg_val, eg_vect = linalg.eig( np.matmul(np.conjugate(array.T), array) )

	# alpha = np.sqrt(np.min(np.absolute(eg_val)))
	# beta = np.sqrt(np.max(np.absolute(eg_val)))

	eg_val, eg_vect = linalg.eig(array)

	alpha = np.min(np.absolute(eg_val))
	beta = np.max(np.absolute(eg_val))

	X = (1/(beta**2 - alpha**2))*( 2*linalg.fractional_matrix_power(array, 2) - (beta**2 + alpha**2)*np.identity(np.shape(array)[0], dtype=np.complex_) )

	temp_sum = np.zeros( np.shape(array), dtype=np.complex_)
	for n in range(N):
		temp_sum += Chebyshev_polynomials_factors(n, N, alpha, beta)*Chebyshev_polynomials_matrix_function(X, n)

	return np.matmul(array, temp_sum)

def formal_sign_function(array):

  array_norm_squared = np.matmul(array, np.conjugate(array.T))
  inverse_array_norm = fractional_matrix_power(array_norm_squared, -0.5)
  sign_function_array = np.matmul(array, inverse_array_norm)

  return sign_function_array


def r(x, alpha, beta):
  return (0.5*(beta**2 + alpha**2) + 0.5*x*(beta**2 - alpha**2))**(-0.5)

def Chebyshev_polynomials_function(x, n):

  Tn_minus1 = 1
  if (n==0):
    return Tn_minus1
  else:
    Tn = x
    for order in range(1, n):
      temp = Tn
      Tn = 2*x*Tn - Tn_minus1
      Tn_minus1 = temp

    return Tn

def Chebyshev_polynomials_factors(n, N, alpha, beta):

  sum = 0.0
  for k in range(1, N+1):
    xk = np.cos((k - 0.5)*np.pi/float(N))
    sum += r(xk, alpha, beta)*Chebyshev_polynomials_function(xk, n)

  # return np.pi/float(N)*sum
  if (n==0):
    factor = 1.0/float(N)
  else:
    factor = 2.0/float(N)
  
  return factor*sum

#   # return np.pi/float(N)*sum
#   if (n == 0):
#     factor = 1.0/float(N)
#   else:
#     factor = 2.0/float(N)
  
#   return factor*sum

# def r(x, alpha, beta):
#   return (0.5*(beta**2 + alpha**2) + 0.5*x*(beta**2 - alpha**2))**(-0.5)

# class OverlapOperator:
#     def __init__(self, lattice_size, lattice_dimensions, fermion_dimensions, gauge_links_phase_values_field_array, laplacian_stencil, derivative_stencil, bare_mass, alpha, beta):
#         self.gauge_links_phase_values_field_array = gauge_links_phase_values_field_array
#         self.laplacian_stencil = laplacian_stencil
#         self.derivative_stencil = derivative_stencil
#         self.bare_mass = bare_mass
#         self.alpha = alpha
#         self.beta = beta
#         self.lattice_size = lattice_size
#         self.lattice_dimensions = lattice_dimensions
#         self.fermion_dimensions = fermion_dimensions
    
#     def reduced_Wilson_operator_function(self, x):
#         return improved_Wilson_operator_linear_function(self.gauge_links_phase_values_field_array, self.laplacian_stencil, self.derivative_stencil, x)

#     def h_vector(self, x):
#         return gamma5_function(self.reduced_Wilson_operator_function(x)) - gamma5_function(x)

#     def X_column(self, x):
#         return (1/(self.beta**2 - self.alpha**2))*( 2*self.h_vector(self.h_vector(x)) - (self.beta**2 + self.alpha**2)*x )

#     def Chebyshev_polynomials_vector_function(self, n, x):
#         Tn_minus1 = x
#         if (n == 0):
#             return Tn_minus1
#         else:
#             Tn = self.X_column(x)
#             for order in range(1, n):
#                 temp = Tn
#                 Tn = 2*self.X_column(Tn) - Tn_minus1
#                 Tn_minus1 = temp

#         return Tn
    
#     def approximate_matrix_sign_column_function(self, x, N=50):
#         vector_sum = np.zeros( (np.shape(x)[0]), dtype=np.complex_)
#         for n in range(N):
#             vector_sum += Chebyshev_polynomials_factors(n, N, self.alpha, self.beta)*self.Chebyshev_polynomials_vector_function(n, x)

#         result = self.h_vector(vector_sum)

#         return result
    
#     def overlap_operator_linear_function(self, x):
#         return x + gamma5_function(self.approximate_matrix_sign_column_function(self.gauge_links_phase_values_field_array, self.laplacian_stencil, self.derivative_stencil, self.alpha, self.beta, x))

#     def massive_overlap_operator_linear_function(self, x):
#         return (1-0.5*self.bare_mass)*self.overlap_operator_linear_function(self.gauge_links_phase_values_field_array, self.laplacian_stencil, self.derivative_stencil, x) + self.bare_mass*x

#     def inverse_overlap_operator_function(self, number_of_rows=4):

#         x0 = np.zeros((self.fermion_dimensions*(self.lattice_size**self.lattice_dimensions),), dtype=np.complex_)
#         psi_vector_field = np.zeros((self.fermion_dimensions*(self.lattice_size**self.lattice_dimensions),), dtype=np.complex_)
#         psi_vector_field[0] = 1.0

#         inverse_overlap_operator = []
#         for next_permutation in range(number_of_rows):
            
#             b_prime = gamma5_function(self.overlap_operator_linear_function(self.gauge_links_phase_values_field_array, self.laplacian_stencil, self.derivative_stencil, gamma5_function(psi_vector_field)))
            
#             inverse_overlap_operator_vector = functional_conjugate_gradient_solution_function(self.gauge_links_phase_values_field_array, self.laplacian_stencil, self.derivative_stencil, b_prime, x0)
            
#             inverse_overlap_operator.append(inverse_overlap_operator_vector)

#             psi_vector_field = np.roll(psi_vector_field, shift=1)

#         inverse_overlap_operator = (np.array(inverse_overlap_operator))

#         return inverse_overlap_operator.T

# def recursive_Chebyshev_polynomials_function(x, n):

#   if (n == 0):
#     return 1
#   elif (n == 1):
#     return x
#   else:
#   	return 2*x*recursive_Chebyshev_polynomials_function(x, n-1) - recursive_Chebyshev_polynomials_function(x, n-2)