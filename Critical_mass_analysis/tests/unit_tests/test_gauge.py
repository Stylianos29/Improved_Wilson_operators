import unittest
import sys

import numpy as np
import pytest
from hypothesis import example, given, assume, strategies as st
from hypothesis.extra.numpy import arrays

sys.path.append('../..')
import custom_library.auxiliary as auxiliary
import custom_library.gauge as gauge


# HELPER FUNCTIONS, GLOBAL CONSTANTS AND COMPOSITE STRATEGIES
COMMON_TYPE_STRATEGIES = st.none() | st.dates() | st.floats() | st.complex_numbers() | st.tuples() | st.text() # Missing st.booleans() and st.integers()

@st.composite
def tuples_of_integers(draw, min_size=0, max_size=10, min_value=0, max_value=10):
    '''Generates variable-sized tuples of integers within an adjustable range.'''

    elements = st.integers(min_value=min_value, max_value=max_value)
    lists_of_various_size = draw(st.lists(elements, min_size=min_size, max_size=max_size))

    return tuple(lists_of_various_size)

#####################################
######## INPUT VALUES TESTING #######
#####################################

# CONSTRUCTORS
@pytest.mark.input_values_exception_raised_test
class TestGaugeLinksFieldConstructor(unittest.TestCase):
  '''Tests exceptions during object instantiation with out-of-range or of incorrect-type input values for the GaugeLinksField __init__ constructor.'''

  random_gauge_links_field_phase_values_array = gauge.GaugeLinksField.random_gauge_links_field_phase_values_array_function()
  
  @given(gauge_links_field_array = st.booleans() | st.integers() | COMMON_TYPE_STRATEGIES)
  def test_gauge_links_field_array_input(self, gauge_links_field_array):
    self.assertRaises(AssertionError, lambda gauge_links_field_array: gauge.GaugeLinksField(gauge_links_field_array=gauge_links_field_array, gauge_links_phase_value_field_boolean=True, random_gauge_links_field_boolean=True), gauge_links_field_array)

  @given(gauge_links_phase_value_field_boolean = st.integers() | COMMON_TYPE_STRATEGIES)
  def test_gauge_links_phase_value_field_boolean_input(self, gauge_links_phase_value_field_boolean):
    self.assertRaises(AssertionError, lambda gauge_links_phase_value_field_boolean: gauge.GaugeLinksField(self.random_gauge_links_field_phase_values_array, random_gauge_links_field_boolean=True, gauge_links_phase_value_field_boolean=gauge_links_phase_value_field_boolean), gauge_links_phase_value_field_boolean)

  @given(fermion_dimensions = st.integers(min_value=-5, max_value=10).filter(lambda n: n not in [2, 4]) | st.booleans() | COMMON_TYPE_STRATEGIES)
  def test_fermion_dimensions_input(self, fermion_dimensions):
    self.assertRaises(AssertionError, lambda fermion_dimensions: gauge.GaugeLinksField(self.random_gauge_links_field_phase_values_array, gauge_links_phase_value_field_boolean=True, random_gauge_links_field_boolean=True, fermion_dimensions=fermion_dimensions), fermion_dimensions)

  @given(random_gauge_links_field_boolean = st.text() | st.integers() | st.none() | st.dates() | st.floats() | st.complex_numbers()| st.tuples())
  def test_random_gauge_links_field_boolean_input(self, random_gauge_links_field_boolean):
    self.assertRaises(AssertionError, lambda random_gauge_links_field_boolean: gauge.GaugeLinksField(self.random_gauge_links_field_phase_values_array, gauge_links_phase_value_field_boolean=True, random_gauge_links_field_boolean=random_gauge_links_field_boolean), random_gauge_links_field_boolean)

# ATTRIBUTES
@pytest.mark.input_values_exception_raised_test
class TestImmutableAttributes(unittest.TestCase):
  '''Tests if exception are raised when attempting to modify the fundamental immutable parameters.'''

  @given(gauge_theory_label = st.integers())
  def test_gauge_theory_label_input(self, gauge_theory_label):
    test_gauge_links_field_object = gauge.GaugeLinksField.from_lattice_shape_with_random_gauge_links_field()

    with self.assertRaises(auxiliary.ReadOnlyAttributeError):
      test_gauge_links_field_object.gauge_theory_label = gauge_theory_label

  @given(gauge_links_field_array = arrays(dtype=np.complex_, shape=(9,9,2)))
  def test_gauge_links_field_array_input(self, gauge_links_field_array):
    test_gauge_links_field_object = gauge.GaugeLinksField.from_lattice_shape_with_random_gauge_links_field()

    with self.assertRaises(auxiliary.ReadOnlyAttributeError):
      test_gauge_links_field_object.gauge_links_field_array = gauge_links_field_array

  @given(gauge_links_phase_values_field_array = arrays(dtype=np.complex_, shape=(9,9,2)))
  def test_gauge_links_phase_values_field_array_input(self, gauge_links_phase_values_field_array):
    test_gauge_links_field_object = gauge.GaugeLinksField.from_lattice_shape_with_random_gauge_links_field()

    with self.assertRaises(auxiliary.ReadOnlyAttributeError):
      test_gauge_links_field_object.gauge_links_phase_values_field_array = gauge_links_phase_values_field_array

  @given(gauge_links_phase_value_field_boolean = st.booleans())
  def test_gauge_links_phase_value_field_boolean_input(self, gauge_links_phase_value_field_boolean):
    test_gauge_links_field_object = gauge.GaugeLinksField.from_lattice_shape_with_random_gauge_links_field()

    with self.assertRaises(auxiliary.ReadOnlyAttributeError):
      test_gauge_links_field_object.gauge_links_phase_value_field_boolean = gauge_links_phase_value_field_boolean

  @given(random_gauge_links_field_boolean = st.booleans())
  def test_random_gauge_links_field_boolean_input(self, random_gauge_links_field_boolean):
    test_gauge_links_field_object = gauge.GaugeLinksField.from_lattice_shape_with_random_gauge_links_field()

    with self.assertRaises(auxiliary.ReadOnlyAttributeError):
      test_gauge_links_field_object.random_gauge_links_field_boolean = random_gauge_links_field_boolean

# PUBLIC METHODS
@pytest.mark.input_values_exception_raised_test
class TestPublicMethodRandomGaugeArrayGenerator(unittest.TestCase):
  '''Tests exceptions raised for the class method cls.random_gauge_links_field_phase_values_array_function() method for out-of-range or of incorrect-type input values.
  -NOTE: No need to test "lattice_shape" argument since it is passed in a class method of lattice.LatticeStructure class.'''

  @given(gauge_theory_label = st.integers(min_value=-5, max_value=10).filter(lambda n: n not in [1, 2, 3]) | st.booleans() | COMMON_TYPE_STRATEGIES)
  def test_gauge_theory_label_input(self, gauge_theory_label):
    self.assertRaises(AssertionError, lambda gauge_theory_label: gauge.GaugeLinksField.random_gauge_links_field_phase_values_array_function(gauge_theory_label=gauge_theory_label), gauge_theory_label)

  @pytest.mark.skip
  @given(random_generator_seed = st.booleans() | st.booleans() | COMMON_TYPE_STRATEGIES)
  def test_random_generator_seed_input(self, random_generator_seed):
    self.assertRaises(AssertionError, lambda random_generator_seed: gauge.GaugeLinksField.random_gauge_links_field_phase_values_array_function(random_generator_seed=random_generator_seed), random_generator_seed)
  
  @given(random_phase_values_range = st.floats(max_value=0, exclude_max=True) | st.floats(min_value=np.pi/2, exclude_min=True) | st.integers().filter(lambda n: (n < 0) or (n > int(np.pi/2))) | st.complex_numbers() | st.none() | st.dates() | st.tuples() | st.text() | st.booleans())
  def test_random_phase_values_range_input(self, random_phase_values_range):
    self.assertRaises(AssertionError, lambda random_phase_values_range: gauge.GaugeLinksField.random_gauge_links_field_phase_values_array_function(random_phase_values_range=random_phase_values_range), random_phase_values_range)

#####################################
####### OUTPUT VALUES TESTING #######
#####################################

@pytest.mark.output_values_replication_test
@pytest.mark.parametrize("random_phase_values_range, expected_output", [(0, np.zeros((9,9,2)))])
def test_lattice_coordinates_vectors_addition(random_phase_values_range, expected_output):
  '''-NOTE: This test parametrization assumes lattice shape: (9,9).'''

  tested_output = gauge.GaugeLinksField.random_gauge_links_field_phase_values_array_function(random_phase_values_range=random_phase_values_range)

  assert np.all(tested_output == expected_output)


@pytest.mark.output_values_replication_test
def test__converting_to_gauge_links_phase_values_field_function():
  test_gauge_links_field_object = gauge.GaugeLinksField.from_lattice_shape_with_random_gauge_links_field()

  gauge_links_field_array = test_gauge_links_field_object.gauge_links_field_array
  gauge_links_phase_values_field_array = test_gauge_links_field_object.gauge_links_phase_values_field_array

  tested_output = np.real(-1.j*np.log(gauge_links_field_array))
  expected_output = gauge_links_phase_values_field_array

  assert np.allclose(tested_output, expected_output)

#####################################
####### PROPERTY-BASED TESTING ######
#####################################


if __name__ == "__main__":

  # Debugging
  test_gauge_links_field_object = gauge.GaugeLinksField.from_lattice_shape_with_random_gauge_links_field()
  print(repr(test_gauge_links_field_object))
  print()
  print(str(test_gauge_links_field_object))

  unittest.main()