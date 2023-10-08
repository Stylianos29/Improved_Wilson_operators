import numpy as np
from numpy.linalg import inv
from numpy.linalg import matrix_power
import sys
import unittest
import pytest

# Custom modules
sys.path.append('../..')
import custom_library.Chebyshev_polynomials as Chebyshev_polynomials


INPUT_VALUES_RANGE = 200.
STEP_VALUE = np.pi

#############################
#     SCALAR FUNCTIONS
#############################

real_scalar_input_values_list = np.arange(-INPUT_VALUES_RANGE/2, INPUT_VALUES_RANGE/2, STEP_VALUE)

@pytest.mark.input_values_test
class TestValidityOfScalarInput(unittest.TestCase):

  def test_scalar_approximate_cosine_function(self):
    input_value = 0 + 1.j*0
    
    with self.assertRaises(AssertionError):
      Chebyshev_polynomials.scalar_approximate_cosine_function(input_value)
  
  def test_scalar_approximate_cosine_function(self):
    real_scalar_input_values = [0., -1]
    
    with self.assertRaises(AssertionError):
      Chebyshev_polynomials.scalar_approximate_cosine_function(*real_scalar_input_values)

    def test_matrix_approximate_cosine_function_non_square_matrix_input(self):
      pass


@pytest.mark.definition_test
@pytest.mark.parametrize("real_scalar_input_values", real_scalar_input_values_list)
def test_scalar_approximate_cosine_function(real_scalar_input_values):
  expression = Chebyshev_polynomials.scalar_approximate_cosine_function(real_scalar_input_values)
  benchmark = np.cos(real_scalar_input_values)

  assert np.isclose(expression, benchmark)


@pytest.mark.definition_test
@pytest.mark.parametrize("real_scalar_input_values", real_scalar_input_values_list)
def test_scalar_approximate_sine_function(real_scalar_input_values):
  expression = Chebyshev_polynomials.scalar_approximate_sine_function(real_scalar_input_values)
  benchmark = np.sin(real_scalar_input_values)

  assert np.isclose(expression, benchmark)


@pytest.mark.property_test
@pytest.mark.parametrize("real_scalar_input_values", real_scalar_input_values_list)
def test_identity_of_scalar_trigonometric_functions(real_scalar_input_values):
  expression = Chebyshev_polynomials.scalar_approximate_cosine_function(real_scalar_input_values)**2 + Chebyshev_polynomials.scalar_approximate_sine_function(real_scalar_input_values)**2
  benchmark = 1.0  

  assert np.isclose(expression, benchmark)


@pytest.mark.definition_test
@pytest.mark.parametrize("real_scalar_input_values", real_scalar_input_values_list)
def test_Chebyshev_polynomials_of_the_first_kind_definition(real_scalar_input_values):
  n = 4

  expression = Chebyshev_polynomials.scalar_Chebyshev_polynomial_term_function(np.cos(real_scalar_input_values), n)
  benchmark = np.cos(n*real_scalar_input_values)

  assert np.isclose(expression, benchmark), 'The definition of the Chebyshev polynomials of the first kind definition is not respected.'


@pytest.mark.definition_test
@pytest.mark.parametrize("real_scalar_input_values", real_scalar_input_values_list)
def test_Chebyshev_polynomials_of_the_first_kind_recursive_definition(real_scalar_input_values):
  n = 2

  expression = Chebyshev_polynomials.scalar_Chebyshev_polynomial_term_recursive_function(np.cos(real_scalar_input_values), n)
  benchmark = np.cos(n*real_scalar_input_values)

  assert np.isclose(expression, benchmark), 'The recursive definition of the Chebyshev polynomials of the first kind definition is not respected.'

@pytest.mark.property_test
@pytest.mark.parametrize("i, j, expected", [(5, 9, 0), (0, 0, 10), (3, 3, 5)])
def test_Chebyshev_polynomials_of_the_first_kind_discrete_orthogonality_condition(i, j, expected):

  # Arbitrary value for N
  N = 10

  #helper function
  def xk(k):
    return np.cos(np.pi/float(N)*(k+0.5))

  orthogonality_condition = 0.
  for k in range(N):
    orthogonality_condition += Chebyshev_polynomials.scalar_Chebyshev_polynomial_term_function(xk(k), i)*Chebyshev_polynomials.scalar_Chebyshev_polynomial_term_function(xk(k), j)

  assert np.isclose(orthogonality_condition, expected), 'The discrete orthogonality condition of the Chebyshev polynomials of the first kind definition is not satisfied.'

def test_specific_scalar_Chebyshev_polynomials_expansion():

  # Object instantiation
  alpha = 0.1
  beta = 4.
  Chebyshev_polynomials_expansion = Chebyshev_polynomials.ScalarChebyshevPolynomialsSignFunction(alpha, beta)

  x = 0.55
  N = 300
  expression = Chebyshev_polynomials_expansion(x, N)
  benchmark = Chebyshev_polynomials_expansion.r(x)

  assert np.isclose(expression, benchmark), 'The Chebyshev polynomials expansion of the specific function does not converge to the function value for the passed argument.'

#############################
#     MATRIX FUNCTIONS
#############################

real_matrix_input_values_list = [np.random.rand(3,3), np.random.rand(4,4), np.random.rand(5,5)]

@pytest.mark.input_values_test
class TestValidityOfScalarInput(unittest.TestCase):

  def test_scalar_approximate_cosine_function(self):
    input_value = 0 + 1.j*0
    
    with self.assertRaises(AssertionError):
      Chebyshev_polynomials.scalar_approximate_cosine_function(input_value)
  
  def test_scalar_approximate_cosine_function(self):
    real_scalar_input_values = [0., -1]
    
    with self.assertRaises(AssertionError):
      Chebyshev_polynomials.scalar_approximate_cosine_function(*real_scalar_input_values)

    def test_matrix_approximate_cosine_function_non_square_matrix_input(self):
      pass

@pytest.mark.property_test
@pytest.mark.parametrize("real_matrix_input_values", real_matrix_input_values_list)
def test_matrix_identity_of_matrix_trigonometric_functions(real_matrix_input_values):
  expression = matrix_power(Chebyshev_polynomials.matrix_approximate_cosine_function(real_matrix_input_values), 2) + matrix_power(Chebyshev_polynomials.matrix_approximate_sine_function(real_matrix_input_values), 2)
  benchmark = np.identity(expression.shape[0])

  assert np.all(np.isclose(expression, benchmark)), f'The identity is not respected by {max(expression-benchmark)}'


@pytest.mark.definition_test
@pytest.mark.parametrize("real_matrix_input_values", real_matrix_input_values_list)
def test_matrix_Chebyshev_polynomials_of_the_first_kind_definition(real_matrix_input_values):
  n = 4

  expression = Chebyshev_polynomials.matrix_Chebyshev_polynomial_term_function(Chebyshev_polynomials.matrix_approximate_cosine_function(real_matrix_input_values), n)
  benchmark = Chebyshev_polynomials.matrix_approximate_cosine_function(n*real_matrix_input_values)

  assert np.all(np.isclose(expression, benchmark)), 'The definition of the Chebyshev polynomials of the first kind definition is not respected.'


@pytest.mark.definition_test
@pytest.mark.parametrize("real_matrix_input_values", real_matrix_input_values_list)
def test_matrix_Chebyshev_polynomials_of_the_first_kind_recursive_definition(real_matrix_input_values):
  n = 4

  expression = Chebyshev_polynomials.matrix_Chebyshev_polynomial_term_recursive_function(Chebyshev_polynomials.matrix_approximate_cosine_function(real_matrix_input_values), n)
  benchmark = Chebyshev_polynomials.matrix_approximate_cosine_function(n*real_matrix_input_values)

  assert np.all(np.isclose(expression, benchmark)), 'The recursive definition of the Chebyshev polynomials of the first kind definition is not respected.'

# def test_temp():

#   A = np.random.rand(5,5) + 1.j*np.random.rand(5,5)
#   B = np.matmul(np.transpose(np.conjugate(A)), A)

#   sign_function_of = Chebyshev_polynomials.MatrixChebyshevPolynomialsSignFunction()

#   C = sign_function_of(B, N=80)

#   D = Chebyshev_polynomials.matrix_formal_sign_function_function(B)

#   print(np.max(np.identity(B.shape[0]) - matrix_power(D,2)))

#   print(str(sign_function_of))

#   return np.max(C - D)

#############################
#     VECTOR FUNCTIONS
#############################


if __name__ == "__main__":

  unittest.main()
