'''This module contains the ChebyshevPolynomialsSignFunction class definition along with a number of functions solely included for testing purposes. Scalar and matrix functions for constructing Chebyshev polynomial terms of the first kind and an approximate matrix sign.'''

from typing import Any
import numpy as np
from numpy.linalg import matrix_power
from math import factorial
from scipy.linalg import fractional_matrix_power
from scipy import linalg


#############################
#     SCALAR FUNCTIONS
#############################

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

class ScalarChebyshevPolynomialsSignFunction:

  def __init__(self, alpha, beta):

    assert isinstance(alpha, float) and (alpha > 0), ''
    self.alpha = alpha

    assert isinstance(beta, float) and (beta > 0), ''
    self.beta = beta

  def __call__(self, x, N):
  
    assert isinstance(x, float) and (x >= -1.) and (x <= 1.), 'Input argument must be a real number in the range [-1., 1.].'

    assert isinstance(N, int) and (N >= 0), 'The total number of terms parameter N must be a non-negative integer.'
    self.N = N

    Chebyshev_polynomials_expansion_sum = 0.
    for n in range(self.N):
      Chebyshev_polynomials_expansion_sum += self.Chebyshev_polynomials_summation_factors(n)*scalar_Chebyshev_polynomial_term_function(x, n)

    return Chebyshev_polynomials_expansion_sum 

  def r(self, x):
    return (0.5*(self.beta**2 + self.alpha**2) + 0.5*x*(self.beta**2 - self.alpha**2))**(-0.5)
  
  def xk(self, k):
    return np.cos(np.pi/self.N*(k+0.5))

  def Chebyshev_polynomials_summation_factors(self, n):

    sum = 0.0
    for k in range(self.N):
      sum += self.r(self.xk(k))*scalar_Chebyshev_polynomial_term_function(self.xk(k), n)

    pre_factor = 2
    if (n == 0):
      pre_factor = 1
    
    return pre_factor/float(self.N)*sum


class new_ScalarChebyshevPolynomialsSignFunction:

  def __init__(self, alpha, beta):

    assert isinstance(alpha, float) and (alpha > 0), ''
    self.alpha = alpha

    assert isinstance(beta, float) and (beta > 0), ''
    self.beta = beta

  def __call__(self, x, N):
  
    assert isinstance(x, float) and (x >= -1.) and (x <= 1.), 'Input argument must be a real number in the range [-1., 1.].'

    assert isinstance(N, int) and (N >= 0), 'The total number of terms parameter N must be a non-negative integer.'
    self.N = N

    # Initializing cache dictionaries
    self.discrete_xk_values_dict = dict()
    self.discrete_rk_values_dict = dict()

    Chebyshev_polynomials_expansion_sum = 0.
    for n in range(self.N):
      Chebyshev_polynomials_expansion_sum += self.Chebyshev_polynomials_summation_factors(n)*scalar_Chebyshev_polynomial_term_function(x, n)

    return Chebyshev_polynomials_expansion_sum 

  def r(self, x):
    return (0.5*(self.beta**2 + self.alpha**2) + 0.5*x*(self.beta**2 - self.alpha**2))**(-0.5)
  
  def xk(self, k):

    if k in self.discrete_xk_values_dict:
      return self.discrete_xk_values_dict[k]

    self.discrete_xk_values_dict[k] = np.cos(np.pi/self.N*(k+0.5))

    return self.discrete_xk_values_dict[k]

  def Chebyshev_polynomials_summation_factors(self, n):

    # helper function
    def rk(k):
      if k in self.discrete_rk_values_dict:
        return self.discrete_rk_values_dict[k]
      
      self.discrete_rk_values_dict[k] = self.r(self.xk(k))

      return self.discrete_rk_values_dict[k]
    
    expansion_sum = 0.0
    for k in range(self.N):
      expansion_sum += rk(k)*scalar_Chebyshev_polynomial_term_function(self.xk(k), n)

    pre_factor = 2
    if (n == 0):
      pre_factor = 1
    
    return pre_factor/float(self.N)*expansion_sum

#############################
#     MATRIX FUNCTIONS
#############################

# MATRIX TRIGONOMETRIC FUNCTIONS

def matrix_approximate_cosine_function(X, n=20):
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


def matrix_approximate_sine_function(X, n=20):
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

  Tn_minus1, Tn = np.identity(X.shape[0]), X
  for _ in range(1, n):
    Tn_minus1, Tn = Tn, 2*np.matmul(X,Tn) - Tn_minus1
  
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
  
  cache_dictionary[n] = 2*np.matmul(X, matrix_Chebyshev_polynomial_term_recursive_function(X, n-1)) - matrix_Chebyshev_polynomial_term_recursive_function(X, n-2)

  return cache_dictionary[n]

# SIGN FUNCTIONS

def matrix_formal_sign_function_function(input_matrix):
  '''Implementation of the formal definition of the matrix sign function.'''

  input_matrix = np.array(input_matrix)
  assert (input_matrix.ndim == 2) and (input_matrix.shape[0] == input_matrix.shape[1]), 'Input matrix must be a 2-dimensional square array.'

  input_matrix_norm_squared = np.matmul(input_matrix, np.conjugate(input_matrix.T))
  inverse_of_input_matrix_norm_squared = fractional_matrix_power(input_matrix_norm_squared, -0.5)
  sign_function_of_input_matrix = np.matmul(input_matrix, inverse_of_input_matrix_norm_squared)

  return sign_function_of_input_matrix


def matrix_approximate_sign_function(array, N=50):
  '''Input: '''

  eg_val, _ = linalg.eig( np.matmul(np.conjugate(array.T), array) )

  alpha = np.sqrt(np.min(np.absolute(eg_val)))
  beta = np.sqrt(np.max(np.absolute(eg_val)))

  X = (1/(beta**2 - alpha**2))*( 2*np.matmul(np.conjugate(array.T), array) - (beta**2 + alpha**2)*np.identity(np.shape(array)[0], dtype=np.complex_) )

  temp_sum = np.zeros( np.shape(array), dtype=np.complex_)
  for n in range(N):
    temp_sum += Chebyshev_polynomials_factors(n, N, alpha, beta)*matrix_Chebyshev_polynomial_term_function(X, n)

  return np.matmul(array, temp_sum)

#############################
#     VECTOR FUNCTIONS
#############################

class MatrixChebyshevPolynomialsSignFunction(ScalarChebyshevPolynomialsSignFunction):

  def __call__(self, H, N=50):
    
    H = np.array(H)
    
    assert (H.ndim == 2) and (H.shape[0] == H.shape[1]), 'Input matrix must be a 2-dimensional square array.'
    assert np.all(np.isclose(H, np.transpose(np.conjugate(H)))), 'Input array must be complex Hermitian.'
    self.H = H

    assert isinstance(N, int) and (N >= 0), 'The number of Chebyshev polynomials terms N must be a non-negative integer.'
    self.N = N

    self.alpha, self.beta = self. extreme_values_of_the_eigenvalues_spectrum()

    return self.matrix_approximate_sign_function()

  def __str__(self) -> str:
    return f'An input matrix of shape {self.H.shape} was passed.'

  def Chebyshev_polynomials_factors(self, n):

    inverse_N_factor = 1/float(self.N)

    sum = 0.0
    for k in range(0, self.N):
      xk = np.cos((k + 0.5)*np.pi*inverse_N_factor)
      sum += r(xk, self.alpha, self.beta)*scalar_Chebyshev_polynomial_term_function(xk, n)

    factor = 2
    if (n == 0):
      factor = 1

    return factor*inverse_N_factor*sum

  def extreme_values_of_the_eigenvalues_spectrum(self):
    eg_val, _ = linalg.eig(matrix_power(self.H, 2))

    return np.sqrt(np.min(eg_val)), np.sqrt(np.max(eg_val))
  
  def matrix_approximate_sign_function(self):

    X = (1/(self.beta**2 - self.alpha**2))*( 2*matrix_power(self.H, 2) - (self.beta**2 + self.alpha**2)*np.identity((self.H).shape[0], dtype=np.complex_) )

    temp_sum = np.zeros_like(self.H)
    for n in range(self.N):
      temp_sum += Chebyshev_polynomials_factors(n, self.N, self.alpha, self.beta)*matrix_Chebyshev_polynomial_term_function(X, n)

    return np.matmul(self.H, temp_sum)


##################################
# Functions kept to avoid dependency conflicts with other scripts, they need to be removed.

def Chebyshev_polynomials_factors(n, N, alpha, beta):

  sum = 0.0
  for k in range(1, N+1):
    xk = np.cos((k - 0.5)*np.pi/float(N))
    sum += r(xk, alpha, beta)*scalar_Chebyshev_polynomial_term_function(xk, n)

  if (n==0):
    factor = 1/float(N)
  else:
    factor = 2/float(N)
  
  return factor*sum


def Chebyshev_polynomials_column_function(n, x, b):
  '''???????????'''

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


def r(x, alpha, beta):
  return (0.5*(beta**2 + alpha**2) + 0.5*x*(beta**2 - alpha**2))**(-0.5)
