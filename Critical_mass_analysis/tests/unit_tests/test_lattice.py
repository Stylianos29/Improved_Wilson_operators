import unittest
import sys

import pytest
from hypothesis import example, given, assume, strategies as st

sys.path.append('../..')
import custom_library.auxiliary as auxiliary
import custom_library.lattice as lattice


#####################################
######## INPUT VALUES TESTING #######
#####################################

# CONSTRUCTORS
@pytest.mark.input_values_exception_raised_test
class TestLatticeStructureConstructor(unittest.TestCase):
  '''Tests exceptions during object instantiation with out-of-range or of incorrect-type input values for the LatticeStructure __init__ constructor.'''
  
  @given(lattice_size = st.integers().filter(lambda n: not n >= 9) | auxiliary.generate_common_strategies_apart_from(['integer']))
  def test_lattice_size_input(self, lattice_size):
    self.assertRaises(AssertionError, lambda lattice_size: lattice.LatticeStructure(lattice_size=lattice_size), lattice_size)
 
  @given(lattice_dimensions = st.integers(min_value=-5, max_value=10).filter(lambda n: n not in [1, 2, 3, 4]) | auxiliary.generate_common_strategies_apart_from(['integer']))
  def test_lattice_dimensions_input(self, lattice_dimensions):
    self.assertRaises(AssertionError, lambda lattice_dimensions: lattice.LatticeStructure(lattice_dimensions=lattice_dimensions), lattice_dimensions)

  @given(fermion_dimensions = st.integers(min_value=-5, max_value=10).filter(lambda n: n not in [2, 4]) | auxiliary.generate_common_strategies_apart_from(['integer']))
  def test_fermion_dimensions_input(self, fermion_dimensions):
    self.assertRaises(AssertionError, lambda fermion_dimensions: lattice.LatticeStructure(fermion_dimensions=fermion_dimensions), fermion_dimensions)

  @given(temporal_axis_size = st.integers().filter(lambda n: not n >= 9) | auxiliary.generate_common_strategies_apart_from(['integer', 'none'])) # Default value: None
  def test_temporal_axis_size_input(self, temporal_axis_size):
    self.assertRaises(AssertionError, lambda temporal_axis_size: lattice.LatticeStructure(temporal_axis_size=temporal_axis_size), temporal_axis_size)


# ATTRIBUTES
@pytest.mark.input_values_exception_raised_test
class TestImmutableAttributes(unittest.TestCase):
  '''Tests if exception are raised when attempting to modify the fundamental immutable parameters of the lattice by any values.'''

  @given(lattice_size = st.integers())
  def test_lattice_size_input(self, lattice_size):
    test_lattice_object = lattice.LatticeStructure()

    with self.assertRaises(auxiliary.ReadOnlyAttributeError):
      test_lattice_object.lattice_size = lattice_size

  @given(lattice_dimensions = st.integers())
  def test_lattice_dimensions_input(self, lattice_dimensions):
    test_lattice_object = lattice.LatticeStructure()

    with self.assertRaises(auxiliary.ReadOnlyAttributeError):
      test_lattice_object.lattice_dimensions = lattice_dimensions

  @given(fermion_dimensions = st.integers())
  def test_fermion_dimensions_input(self, fermion_dimensions):
    test_lattice_object = lattice.LatticeStructure()

    with self.assertRaises(auxiliary.ReadOnlyAttributeError):
      test_lattice_object.fermion_dimensions = fermion_dimensions

  @given(lattice_shape = auxiliary.tuples_of_numbers('int', ))
  def test_lattice_shape_input(self, lattice_shape):
    test_lattice_object = lattice.LatticeStructure()

    with self.assertRaises(auxiliary.ReadOnlyAttributeError):
      test_lattice_object.lattice_shape = lattice_shape


# PUBLIC METHODS
@pytest.mark.input_values_exception_raised_test
class TestPublicMethodVectorsAddition(unittest.TestCase):
  '''Tests exceptions raised for the public .lattice_coordinates_vectors_addition() method for out-of-range or of incorrect-type input values.'''
  
  test_lattice_object = lattice.LatticeStructure()

  @given(tuple_a = auxiliary.generate_common_strategies_apart_from(['tuples of integers']), tuple_b = auxiliary.generate_common_strategies_apart_from(['tuples of integers']))
  def test_lattice_coordinates_vectors_addition_with_vectors_of_incorrect_type(self, tuple_a, tuple_b):
    with self.assertRaises(AssertionError):
      self.test_lattice_object.lattice_coordinates_vectors_addition(tuple_a, tuple_b)

  @given(tuple_a = auxiliary.tuples_of_numbers('int', min_value=0, max_value=8), tuple_b = auxiliary.tuples_of_numbers('int', min_value=0, max_value=8))
  def test_lattice_coordinates_vectors_addition_with_vectors_of_different_size(self, tuple_a, tuple_b):
    assume(len(tuple_a) != len(tuple_b))

    with self.assertRaises(AssertionError):
      self.test_lattice_object.lattice_coordinates_vectors_addition(tuple_a, tuple_b)

  @given(tuple_size=st.integers(min_value=1, max_value=10), data=st.data())
  def test_lattice_coordinates_vectors_addition_with_vectors_of_different_size_than_the_lattice_dimensions(self, data, tuple_size):
    tuple_a = data.draw(auxiliary.tuples_of_numbers('int', min_size=tuple_size, max_size=tuple_size, min_value=0, max_value=8), "tuple_a")
    tuple_b = data.draw(auxiliary.tuples_of_numbers('int', min_size=tuple_size, max_size=tuple_size, min_value=0, max_value=8), "tuple_b")

    assume(len(tuple_a) != self.test_lattice_object.lattice_dimensions)

    with self.assertRaises(AssertionError):
      self.test_lattice_object.lattice_coordinates_vectors_addition(tuple_a, tuple_b)

  @given(data=st.data())
  def test_lattice_coordinates_vectors_addition_with_vectors_of_different_size_than_the_lattice_dimensions(self, data):
    tuple_size = self.test_lattice_object.lattice_dimensions

    tuple_a = data.draw(auxiliary.tuples_of_numbers('int', min_size=tuple_size, max_size=tuple_size, min_value=9, max_value=20), "tuple_a")
    tuple_b = data.draw(auxiliary.tuples_of_numbers('int', min_size=tuple_size, max_size=tuple_size, min_value=9, max_value=20), "tuple_b")

    with self.assertRaises(AssertionError):
      self.test_lattice_object.lattice_coordinates_vectors_addition(tuple_a, tuple_b)


@pytest.mark.input_values_exception_raised_test
class TestPublicMethodTurningFundamentalParameters(unittest.TestCase):
  '''Tests exceptions raised for the public .turning_fundamental_parameters_to_lattice_shape() class method for out-of-range or of incorrect-type input values.'''

# lattice_size, lattice_dimensions, *, temporal_axis_size=None

  @given(lattice_size = st.integers().filter(lambda n: not n >= 9) | auxiliary.generate_common_strategies_apart_from(['integer']))
  def test_lattice_size_input(self, lattice_size):
    self.assertRaises(AssertionError, lambda lattice_size: lattice.LatticeStructure.turning_fundamental_parameters_to_lattice_shape(lattice_size=lattice_size), lattice_size)
  
  @given(lattice_dimensions = st.integers(min_value=-5, max_value=10).filter(lambda n: n not in [1, 2, 3, 4]) | auxiliary.generate_common_strategies_apart_from(['integer']))
  def test_lattice_dimensions_input(self, lattice_dimensions):
    self.assertRaises(AssertionError, lambda lattice_dimensions: lattice.LatticeStructure.turning_fundamental_parameters_to_lattice_shape(lattice_dimensions=lattice_dimensions), lattice_dimensions)
  
  @given(temporal_axis_size = st.integers().filter(lambda n: not n >= 9) | auxiliary.generate_common_strategies_apart_from(['integer', 'none'])) # Default value: None
  def test_temporal_axis_size_input(self, temporal_axis_size):
    self.assertRaises(AssertionError, lambda temporal_axis_size: lattice.LatticeStructure.turning_fundamental_parameters_to_lattice_shape(temporal_axis_size=temporal_axis_size), temporal_axis_size)


@pytest.mark.input_values_exception_raised_test
class TestPublicMethodTurningLatticeShape(unittest.TestCase):
  '''Tests exceptions raised for the public .turning_lattice_shape_to_fundamental_parameters() class method for out-of-range or of incorrect-type input values.'''

  @given(lattice_shape = auxiliary.tuples_of_numbers('int', min_size=5) | auxiliary.tuples_of_numbers('int', min_size=1, max_size=4, max_value=8) | auxiliary.tuples_of_numbers('float') | auxiliary.generate_common_strategies_apart_from())
  @example(tuple())
  def test_lattice_shape_input(self, lattice_shape):
    self.assertRaises(AssertionError, lambda lattice_shape: lattice.LatticeStructure.turning_lattice_shape_to_fundamental_parameters(lattice_shape=lattice_shape), lattice_shape)

#####################################
####### OUTPUT VALUES TESTING #######
#####################################

@pytest.mark.output_values_replication_test
@given(lattice_shape = auxiliary.tuples_of_numbers('int', min_size=1, max_size=4, min_value=9, max_value=15))
def test_turning_lattice_shape_to_fundamental_parameters(lattice_shape):
  '''Tests temporal_axis_size=None output'''

  assume(lattice_shape[0] == lattice_shape[-1])

  _, _, temporal_axis_size = lattice.LatticeStructure.turning_lattice_shape_to_fundamental_parameters(lattice_shape)
  
  tested_output = temporal_axis_size
  expected_output = None

  assert tested_output == expected_output


@pytest.mark.output_values_replication_test
@pytest.mark.parametrize("tuple_a, tuple_b, expected_output",
      [# positive directions
      ((0,0), (0,0), (0,0)),
      ((0,8), (0,1), (0,0)),
      ((8,0), (1,0), (0,0)),
      ((8,8), (1,1), (0,0)),
      # negative directions
      ((0,0), (0,-1), (0,8)),
      ((0,0), (-1,0), (8,0)),
      ((0,0), (-1,-1), (8,8))]
)
def test_lattice_coordinates_vectors_addition(tuple_a, tuple_b, expected_output):
  '''-NOTE: This test parametrization assumes lattice shape: (9,9).'''

  test_lattice_object = lattice.LatticeStructure()

  tested_output = test_lattice_object.lattice_coordinates_vectors_addition(tuple_a, tuple_b)

  assert tested_output == expected_output

@pytest.mark.output_values_replication_test
@given(lattice_shape = auxiliary.tuples_of_numbers('int', min_size=1,max_size=4, min_value=9, max_value=15))
def test_turning_lattice_shape_to_fundamental_parameters(lattice_shape):
  fundamental_parameters = lattice.LatticeStructure.turning_lattice_shape_to_fundamental_parameters(lattice_shape=lattice_shape)

  tested_output = lattice.LatticeStructure.turning_fundamental_parameters_to_lattice_shape(fundamental_parameters)

  lattice_shape = (lattice_shape[0],) + (lattice_shape[-1],) * (len(lattice_shape) - 1)

  expected_output = lattice_shape

  assert tested_output == expected_output

@pytest.mark.output_values_replication_test
@given(lattice_size = st.integers(min_value=9, max_value=15), lattice_dimensions = st.integers(min_value=2, max_value=4), temporal_axis_size = st.none()  | st.integers(min_value=9, max_value=15))
def test_turning_lattice_shape_to_fundamental_parameters(lattice_size, lattice_dimensions, temporal_axis_size):
  fundamental_parameters = lattice_size, lattice_dimensions, temporal_axis_size
  
  lattice_shape = lattice.LatticeStructure.turning_fundamental_parameters_to_lattice_shape(lattice_size, lattice_dimensions, temporal_axis_size=temporal_axis_size)

  if (lattice_shape[0] ==  lattice_shape[-1]):
    temporal_axis_size = None
  fundamental_parameters = lattice_size, lattice_dimensions, temporal_axis_size

  tested_output = lattice.LatticeStructure.turning_lattice_shape_to_fundamental_parameters(lattice_shape)

  expected_output = fundamental_parameters

  assert tested_output == expected_output

#####################################
####### PROPERTY-BASED TESTING ######
#####################################

@pytest.mark.property_based_test
@given(lattice_dimensions=st.integers(min_value=1, max_value=4), data=st.data())
def test_lattice_coordinates_vectors_addition_reversibility_property(lattice_dimensions, data):
  '''Tests whether the opposite of the input vector b can be used to reproduce a by adding it to the resulting vector c of the addition of a and b, or in other words, it tests whether a=c+(-b), with c=a+b.'''

  test_lattice_object = lattice.LatticeStructure(lattice_dimensions=lattice_dimensions)
  size = test_lattice_object.lattice_dimensions

  tuple_a = data.draw(auxiliary.tuples_of_numbers('int', min_size=size, max_size=size, min_value=0, max_value=8), "tuple_a")
  tuple_b = data.draw(auxiliary.tuples_of_numbers('int', min_size=size, max_size=size, min_value=0, max_value=8), "tuple_b")

  resulting_tuple = test_lattice_object.lattice_coordinates_vectors_addition(tuple_a, tuple_b)

  reversed_sign_tuple_b = tuple(-component for component in tuple_b)

  tested_expression = test_lattice_object.lattice_coordinates_vectors_addition(resulting_tuple, reversed_sign_tuple_b)

  benchmark_expression = tuple_a

  assert tested_expression == benchmark_expression


if __name__ == "__main__":

  # Debugging
  test_lattice_object = lattice.LatticeStructure()
  print(repr(test_lattice_object))
  print()
  print(str(test_lattice_object))

  unittest.main()