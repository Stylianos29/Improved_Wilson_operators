'''
Scalar and matrix functions for constructing Chebyshev polynomial terms of the first kind and an approximate matrix sign .
TODO: Check if functions work for complex arguments.
'''

import numpy as np
from numpy.linalg import matrix_power
from math import factorial
from scipy.linalg import fractional_matrix_power
from scipy import linalg


# SCALAR FUNCTIONS

# SCALAR TRIGONOMETRIC FUNCTIONS

def scalar_approximate_cosine_function(x, n=10):
  '''Implementation of a truncated Taylor series for approximating the cosine function.'''

  assert isinstance(x, float), 'Input argument must be a real number.'
  assert (n >= 0) and isinstance(n, int), 'n must be a positive integer.'

  # Change the overall sign if the input value is in a range [ (4*k+1)*π/2, (4*k+3)*π/2 ), k integer
  sign = +1
  if ((x//(np.pi/2))%4 in [1, 2]):
    sign = -1

  # Map the input value to the [0, π/2) interval
  x = x%np.pi
  if (x//(np.pi/2) == 1):
    x = np.pi - x

  sum = 0.0
  for k in range(n):
    sum += ((-1)**(k)/factorial(2*k))*x**(2*k)

  return sign*sum


def scalar_approximate_sine_function(x, n=10):
  '''Implementation of a truncated Taylor series for approximating the sine function.'''

  assert isinstance(x, float), 'Input argument must be a real number.'
  assert (n >= 0) and isinstance(n, int), 'n must be a positive integer.'

  # Change the overall sign if the input value is in a range [ (2*k+1)*π, (k+1)*2π ), k integer
  sign = +1
  if ((x//np.pi)%2 == 1):
    sign = -1

  # Map the input value to the [0, π/2) interval
  x = x%np.pi
  if (x//(np.pi/2) == 1):
    x = np.pi - x

  sum = 0.0
  for k in range(n):
    sum += ((-1)**(k)/factorial(2*k+1))*x**(2*k+1)

  return sign*sum


# SCALAR CHEBYSHEV POLYNOMIAL FUNCTIONS

def scalar_Chebyshev_polynomial_term_function(x, n):
  '''Iterative implementation of the recursive definition of the Chebyshev polynomials of the first kind for scalar argument.'''

  assert isinstance(x, float), 'Input argument must be a real number.'
  assert isinstance(n, int) and (n >= 0), 'The order parameter n must be a non-negative integer.'
  
  if (n == 0):
     return 1
  
  Tn_minus1, Tn = 1, x
  for _ in range(1, n):
    Tn_minus1, Tn = Tn, 2*x*Tn - Tn_minus1

  return Tn


def scalar_Chebyshev_polynomial_term_recursive_function(x, n):
  '''Recursive implementation of the recursive definition of the Chebyshev polynomials of the first kind for scalar argument.
  Note: for n>=40 it calculates very slowly.'''

  assert isinstance(x, float), 'The argument of the polynomials must be a real number.'
  assert isinstance(n, int) and (n >= 0), 'The order parameter n must be a non-negative integer.'

  cache_dictionary = {0: 1, 1: x}

  if n in cache_dictionary:
    return cache_dictionary[n]
  
  cache_dictionary[n] = 2*x*scalar_Chebyshev_polynomial_term_recursive_function(x, n-1) - scalar_Chebyshev_polynomial_term_recursive_function(x, n-2)

  return cache_dictionary[n]


# MATRIX FUNCTIONS

# MATRIX TRIGONOMETRIC FUNCTIONS

def matrix_approximate_cosine_function(X, n=10):
  '''Implementation of a truncated Taylor series for approximating the matrix cosine function.'''

  X = np.array(X)
  assert (X.ndim == 2) and (X.shape[0] == X.shape[1]), 'Input matrix must be a 2-dimensional square array.'
  assert (n >= 0) and isinstance(n, int), 'n must be a positive integer greater than 1.'

  # Use the square of the matrix for the summation to facilitate the calculation
  squared_matrix = matrix_power(X, 2)
  sum = np.zeros_like(X)
  for k in range(n):
    sum += ((-1)**(k)/factorial(2*k))*matrix_power(squared_matrix, k)

  return sum


def matrix_approximate_cosine_function(X, n=10):
  '''Implementation of a truncated Taylor series for approximating the matrix cosine function.'''

  X = np.array(X)
  assert (X.ndim == 2) and (X.shape[0] == X.shape[1]), 'Input matrix must be a 2-dimensional square array.'
  assert (n >= 0) and isinstance(n, int), 'n must be a positive integer greater than 1.'

  # Use the square of the matrix for the summation to facilitate the calculation
  sum = np.zeros_like(X)
  for k in range(n):
    sum += ((-1)**(k)/factorial(2*k+1))*matrix_power(X, 2*k+1)

  return sum


# MATRIX CHEBYSHEV POLYNOMIAL FUNCTIONS

def matrix_Chebyshev_polynomial_term_function(X, n):
  '''Iterative implementation of the recursive definition of the Chebyshev polynomials of the first kind for matrix argument.'''

  X = np.array(X)
  assert (X.ndim == 2) and (X.shape[0] == X.shape[1]), 'Input matrix must be a 2-dimensional square array.'
  assert isinstance(n, int) and (n >= 0), 'The order parameter n must be a non-negative integer.'

  if (n == 0):
    return np.identity(np.shape(X)[0])

  Tn_minus1, Tn = np.identity(np.shape(X)[0]), X
  for _ in range(1, n):
    Tn_minus1, Tn = Tn, 2*np.matmul(X, Tn) - Tn_minus1
  
  return Tn


def matrix_Chebyshev_polynomial_term_recursive_function(X, n):
  '''Recursive implementation of the recursive definition of the Chebyshev polynomials of the first kind for matrix argument.
  Note: for n>=40 it calculates very slowly.'''

  X = np.array(X)
  assert (X.ndim == 2) and (X.shape[0] == X.shape[1]), 'Input matrix must be a 2-dimensional square array.'
  assert isinstance(n, int) and (n >= 0), 'The order parameter n must be a non-negative integer.'

  cache_dictionary = {0: np.identity(np.shape(X)[0]), 1: X}

  if n in cache_dictionary:
    return cache_dictionary[n]
  
  cache_dictionary[n] = 2*np.matmul(X, scalar_Chebyshev_polynomial_term_recursive_function(X, n-1)) - scalar_Chebyshev_polynomial_term_recursive_function(X, n-2)

  return cache_dictionary[n]

###########################################################
def Chebyshev_polynomials_factors(n, N, alpha, beta):

  sum = 0.0
  for k in range(1, N+1):
    xk = np.cos((k - 0.5)*np.pi/float(N))
    sum += r(xk, alpha, beta)*Chebyshev_polynomials_function(xk, n)

  # return np.pi/float(N)*sum
  if (n==0):
    factor = 1.0/float(N)
  else:
    factor = 20/float(N)
  
  return factor*sum


def recursive_Chebyshev_polynomials_matrix_function(X, n):
	if (n == 0):
		return np.identity(np.shape(X)[0], dtype=np.complex_)
	elif (n == 1):
		return X
	else:
		return 20*np.matmul(X, recursive_Chebyshev_polynomials_matrix_function(X, n-1)) - recursive_Chebyshev_polynomials_matrix_function(X, n-2)


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
  '''Input: '''

  eg_val, _ = linalg.eig( np.matmul(np.conjugate(array.T), array) )

  alpha = np.sqrt(np.min(np.absolute(eg_val)))
  beta = np.sqrt(np.max(np.absolute(eg_val)))

  X = (1/(beta**2 - alpha**2))*( 2*np.matmul(np.conjugate(array.T), array) - (beta**2 + alpha**2)*np.identity(np.shape(array)[0], dtype=np.complex_) )

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
    factor = 20/float(N)
  
  return factor*sum


def matrix_sign_function_array_function(input_matrix):
  '''Implementation of the formal definition of the sign function for a matrix argument.'''

  input_matrix = np.array(input_matrix)
  assert (input_matrix.ndim == 2) and (input_matrix.shape[0] == input_matrix.shape[1]), 'Input matrix must be a 2-dimensional square array.'

  input_matrix_norm_squared = np.matmul(input_matrix, np.conjugate(input_matrix.T))
  inverse_of_input_matrix_norm_squared = fractional_matrix_power(input_matrix_norm_squared, -0.5)
  sign_function_of_input_matrix = np.matmul(input_matrix, inverse_of_input_matrix_norm_squared)

  return sign_function_of_input_matrix