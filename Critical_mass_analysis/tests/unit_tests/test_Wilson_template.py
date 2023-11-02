import unittest
import pytest
from hypothesis import example, given, assume, strategies as st
import sys

import numpy as np
from numpy.linalg import inv
from numpy.linalg import matrix_power

# Include the project's main directory in the system path
sys.path.append('../..')
import custom_library.auxiliary as auxiliary
import custom_library.constants as constants
import custom_library.Wilson_template as Wilson_template


# Helper functions, constants and composite strategies
UNIT_DIRECTIONS_TUPLES_DICTIONARY = {1: [(1,), (-1,)], 2: [(1, 0), (-1, 0), (0, -1), (0, 1)], 3: [(1, 0, 0), (-1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, 1), (0, 0, -1)], 4: [(1, 0, 0, 0), (-1, 0, 0, 0), (0, -1, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, -1, 0), (0, 0, 0, -1), (0, 0, 0, 1)]}

@st.composite
def generate_fundamental_parameters(draw):
    lattice_size = draw(st.integers(min_value=9, max_value=10))
    lattice_dimensions = draw(st.integers(min_value=1, max_value=2))

    return lattice_size, lattice_dimensions

@st.composite
def generate_unit_direction_tuple(draw, lattice_dimensions):
    unit_directions_tuples_list_index = draw(st.integers(min_value=0, max_value=2*lattice_dimensions-1))
    
    return UNIT_DIRECTIONS_TUPLES_DICTIONARY[lattice_dimensions][unit_directions_tuples_list_index]

@st.composite
def generate_lattice_site_coordinates(draw, fundamental_parameters):
    lattice_size, lattice_dimensions = fundamental_parameters

    lattice_site_coordinates = draw(st.lists(st.integers(min_value=0, max_value=lattice_size-1), min_size=lattice_dimensions, max_size=lattice_dimensions))

    return tuple(lattice_site_coordinates)

@st.composite
def generate_matrix_row_index(draw, lattice_size, exponent, factor=1):
    matrix_row_indices_range = factor*lattice_size**exponent

    matrix_row_index = draw(st.integers(min_value=0, max_value=matrix_row_indices_range-1))

    return matrix_row_index

def construct_StandardWilsonOperator_object(fundamental_parameters=None, lattice_size_boolean=True, lattice_dimensions_boolean=True):
    if (fundamental_parameters is not None):
        lattice_size, lattice_dimensions = fundamental_parameters

    if (not lattice_size_boolean) and lattice_dimensions_boolean:
        return Wilson_template.StandardWilsonOperator(lattice_dimensions=lattice_dimensions)

    elif lattice_size_boolean and (not lattice_dimensions_boolean):
        return Wilson_template.StandardWilsonOperator(lattice_size=lattice_size)

    elif (not lattice_size_boolean) and (not lattice_dimensions_boolean):
        return Wilson_template.StandardWilsonOperator()
    
    return Wilson_template.StandardWilsonOperator(lattice_size=lattice_size, lattice_dimensions=lattice_dimensions)

def test___repr__():
    standard_Wilson_operator = construct_StandardWilsonOperator_object(lattice_size_boolean=False, lattice_dimensions_boolean=False)

    print(repr(standard_Wilson_operator))

def test___str__():
    standard_Wilson_operator = construct_StandardWilsonOperator_object(lattice_size_boolean=False, lattice_dimensions_boolean=False)

    print(str(standard_Wilson_operator))

common_types_strategies = st.none() | st.floats() | st.text() | st.complex_numbers()

#####################################
######## INPUT VALUES TESTING #######
#####################################

@pytest.mark.input_values_exception_raised_test
class TestClassConstructors(unittest.TestCase):
    '''Testing assertion exceptions for the arguments of the Wilson_template.StandardWilsonOperator class constructor.'''

    @given(input_test_value = common_types_strategies | st.integers(max_value=8))
    def test_lattice_size_input_error(self, input_test_value):
        self.assertRaises(AssertionError, Wilson_template.StandardWilsonOperator, lattice_size=input_test_value)

    @given(input_test_value = common_types_strategies | st.integers().filter(lambda num: (num < 1) and (num > 4)))
    def test_lattice_dimensions_input_error(self, input_test_value):
        self.assertRaises(AssertionError, Wilson_template.StandardWilsonOperator, lattice_dimensions=input_test_value)

    @given(input_test_value = common_types_strategies | st.integers().filter(lambda num: (num != 2) and (num != 4)))
    def test_fermion_dimensions_input_error(self, input_test_value):
        self.assertRaises(AssertionError, Wilson_template.StandardWilsonOperator, fermion_dimensions=input_test_value)

    @given(input_test_value = common_types_strategies | st.integers().filter(lambda num: (num < 1) and (num > 3)))
    def test_gauge_theory_label_input_error(self, input_test_value):
        self.assertRaises(AssertionError, Wilson_template.StandardWilsonOperator, gauge_theory_label=input_test_value)

    @given(input_test_value = st.floats() | st.text() | st.complex_numbers() | st.integers(max_value=8)) # NOTE: Default: None
    def test_temporal_axis_size_input_error(self, input_test_value):
        self.assertRaises(AssertionError, Wilson_template.StandardWilsonOperator, temporal_axis_size=input_test_value)

    @given(input_test_value = st.none() | st.integers() | st.text() | st.complex_numbers() | st.floats(max_value=0.0, exclude_max=True))
    def test_effective_bare_mass_value_input_error(self, input_test_value):
        self.assertRaises(AssertionError, Wilson_template.StandardWilsonOperator, effective_bare_mass_value=input_test_value)

'''
TODO: Test methods with "unit_direction_tuple" and "switching_of_gauge_field_boolean" argument.
'''

#####################################
####### OUTPUT VALUES TESTING #######
#####################################

'''All module functions/methods appear in reverse order from that inside the module.'''

@pytest.mark.output_values_replication_test
@given(st.data())
def test__directed_gauge_links_field_function(data):
    '''Testing the U[n; -mu] = U^†[n-mu; -mu] definition.'''

    fundamental_parameters = data.draw(generate_fundamental_parameters(), "fundamental_parameters")
    lattice_size, lattice_dimensions = fundamental_parameters
    
    standard_Wilson_operator = construct_StandardWilsonOperator_object(fundamental_parameters)

    unit_direction_tuple = data.draw(generate_unit_direction_tuple(lattice_dimensions=lattice_dimensions), "unit_direction_tuple")

    lattice_site_coordinates = data.draw(generate_lattice_site_coordinates(fundamental_parameters), "lattice_site_coordinate")

    tested_output = standard_Wilson_operator._directed_gauge_links_field_function(unit_direction_tuple)[lattice_site_coordinates]

    gauge_links_field_array_axis_index = standard_Wilson_operator._mapping_lattice_direction_to_axis_index(unit_direction_tuple)
    
    if (sum(unit_direction_tuple) > 0):
        replicated_output = standard_Wilson_operator.gauge_links_field_array[lattice_site_coordinates][gauge_links_field_array_axis_index]

    else:
        shifted_lattice_site_coordinates = standard_Wilson_operator.lattice_sites_coordinates_addition(lattice_site_coordinates, unit_direction_tuple)

        replicated_output = np.conjugate(standard_Wilson_operator.gauge_links_field_array[shifted_lattice_site_coordinates][gauge_links_field_array_axis_index])

    assert tested_output == replicated_output


@pytest.mark.output_values_replication_test
@pytest.mark.parametrize("lattice_site_coordinates_tuple, operator_matrix_row", [((0, 0), 0), ((8, 8), 80)])
def test__mapping_lattice_sites_coordinates_to_operator_matrix_coordinate(lattice_site_coordinates_tuple, operator_matrix_row):
    standard_Wilson_operator = Wilson_template.StandardWilsonOperator(lattice_dimensions=len(lattice_site_coordinates_tuple))

    tested_output = standard_Wilson_operator._mapping_lattice_sites_coordinates_to_operator_matrix_coordinate(lattice_site_coordinates_tuple)

    replicated_output = operator_matrix_row

    assert tested_output == replicated_output


@pytest.mark.skip
@pytest.mark.output_values_replication_test
@given(st.data())
def test__matrix_coordinates_space_component_of_operator_function(data):
    fundamental_parameters = data.draw(generate_fundamental_parameters(), "fundamental_parameters")

    lattice_size, lattice_dimensions = fundamental_parameters
    
    standard_Wilson_operator = construct_StandardWilsonOperator_object(fundamental_parameters)

    unit_direction_tuple = data.draw(generate_unit_direction_tuple(lattice_dimensions=lattice_dimensions), "unit_direction_tuple")

    matrix_coordinates_space_component_of_operator_array = standard_Wilson_operator._matrix_coordinates_space_component_of_operator_function(unit_direction_tuple)

    matrix_row_index = data.draw(generate_matrix_row_index(lattice_size=lattice_size, exponent=lattice_dimensions), "matrix_row_index")

    temp = standard_Wilson_operator._mapping_lattice_sites_coordinates_to_operator_matrix_coordinate(unit_direction_tuple)

    if unit_direction_tuple == (0, -1):
        temp = 0
    elif unit_direction_tuple == (0, +1):
        temp = +1


    print(temp)

    assume(matrix_row_index != 0)
    assume((matrix_row_index+temp)%lattice_size != 0)

    nonzero_matrix_elements_indices = np.transpose(np.nonzero(matrix_coordinates_space_component_of_operator_array))[matrix_row_index]

    print(unit_direction_tuple)

    print(np.transpose(np.nonzero(matrix_coordinates_space_component_of_operator_array)))

    tested_output = nonzero_matrix_elements_indices[1]

    replicated_output = (nonzero_matrix_elements_indices[0] + standard_Wilson_operator._mapping_lattice_sites_coordinates_to_operator_matrix_coordinate(unit_direction_tuple))%(lattice_size**lattice_dimensions)

    assert tested_output == replicated_output


@pytest.mark.output_values_replication_test
@pytest.mark.parametrize("unit_direction_tuple, axis_index", [((1, 0), 1), ((0, 0, -1), 0), ((0, 0, 0, 1), 0)])
def test__mapping_lattice_direction_to_axis_index(unit_direction_tuple, axis_index):
    standard_Wilson_operator = Wilson_template.StandardWilsonOperator(lattice_dimensions=len(unit_direction_tuple))

    tested_output = standard_Wilson_operator._mapping_lattice_direction_to_axis_index(unit_direction_tuple)

    replicated_output = axis_index

    assert tested_output == replicated_output


# No particular test for ._matrix_directed_gauged_term_of_operator_function() method


@pytest.mark.output_values_replication_test
@given(st.data())
def test__matrix_renormalized_mass_term_of_operator_function(data):
    fundamental_parameters = data.draw( generate_fundamental_parameters(), "fundamental_parameters")

    lattice_size, lattice_dimensions = fundamental_parameters

    standard_Wilson_operator = construct_StandardWilsonOperator_object(fundamental_parameters)

    tested_output = standard_Wilson_operator._matrix_renormalized_mass_term_of_operator_function()

    renormalized_mass_value = standard_Wilson_operator.effective_bare_mass_value + standard_Wilson_operator.lattice_dimensions
    matrix_side_length = standard_Wilson_operator.lattice_size**standard_Wilson_operator.lattice_dimensions*standard_Wilson_operator.fermion_dimensions*standard_Wilson_operator.gauge_theory_label**2
    
    replicated_output = renormalized_mass_value*np.identity(matrix_side_length, dtype=np.complex_)

    matrix_indices_tuple = (data.draw(generate_matrix_row_index(lattice_size=lattice_size, exponent=lattice_dimensions, factor=4), "matrix_row_index"), data.draw(generate_matrix_row_index(lattice_size=lattice_size, exponent=lattice_dimensions, factor=4), "matrix_column_index"))

    assert tested_output[matrix_indices_tuple] == replicated_output[matrix_indices_tuple]


@pytest.mark.output_values_replication_test
@given(fundamental_parameters = generate_fundamental_parameters())
def test__list_of_unit_direction_tuples_function(fundamental_parameters):
    standard_Wilson_operator = construct_StandardWilsonOperator_object(fundamental_parameters, lattice_size_boolean=False)

    tested_output = standard_Wilson_operator._list_of_unit_direction_tuples_function()

    replicated_output = UNIT_DIRECTIONS_TUPLES_DICTIONARY[standard_Wilson_operator.lattice_dimensions]

    assert tested_output == replicated_output


def test_eigenvalues_spectrum_function():
    assert True


@pytest.mark.output_values_replication_test
@given(fundamental_parameters = generate_fundamental_parameters())
def test_matrix_standard_Wilson_operator_function_with_gauge_field_switched_off(fundamental_parameters):
    standard_Wilson_operator = construct_StandardWilsonOperator_object(fundamental_parameters, lattice_dimensions_boolean=False)

    _, lattice_dimensions = fundamental_parameters
    
    assume(lattice_dimensions == 2)

    tested_output = standard_Wilson_operator.matrix_standard_Wilson_operator_function(switching_of_gauge_field_boolean=True)

    replicated_output = standard_Wilson_operator._matrix_renormalized_mass_term_of_operator_function()

    for unit_direction_tuple in UNIT_DIRECTIONS_TUPLES_DICTIONARY[2]:
        gamma_matrix_sign = sum(unit_direction_tuple)
        gamma_matrix_index = standard_Wilson_operator._mapping_lattice_direction_to_axis_index(unit_direction_tuple)

        replicated_output += -0.5*np.kron(standard_Wilson_operator._matrix_coordinates_space_component_of_operator_function(unit_direction_tuple), np.identity(standard_Wilson_operator.fermion_dimensions) - gamma_matrix_sign*constants.gamma_matrices[gamma_matrix_index])

    assert np.all(tested_output == replicated_output)


@pytest.mark.output_values_replication_test
@given(fundamental_parameters = generate_fundamental_parameters())
def test_matrix_standard_Wilson_operator_function_for_only_2_lattice_dimensions(fundamental_parameters):
    standard_Wilson_operator = construct_StandardWilsonOperator_object(fundamental_parameters, lattice_dimensions_boolean=False)

    _, lattice_dimensions = fundamental_parameters
    
    assume(lattice_dimensions == 2)

    tested_output = standard_Wilson_operator.matrix_standard_Wilson_operator_function()

    replicated_output = standard_Wilson_operator._matrix_renormalized_mass_term_of_operator_function()

    gauged_terms_sum = np.kron(np.einsum("ij,i->ij", standard_Wilson_operator._matrix_coordinates_space_component_of_operator_function((0, +1)), (standard_Wilson_operator._gauge_links_field_array).take(indices=0, axis=-1).reshape(-1) ), np.identity(standard_Wilson_operator.fermion_dimensions) - constants.gamma_matrices[0]) + np.kron(np.einsum("ij,i->ij", standard_Wilson_operator._matrix_coordinates_space_component_of_operator_function((0, -1)), np.conjugate(np.roll((standard_Wilson_operator._gauge_links_field_array).take(indices=0, axis=-1), axis=1, shift=+1).reshape(-1)) ), np.identity(standard_Wilson_operator.fermion_dimensions) + constants.gamma_matrices[0]) + np.kron(np.einsum("ij,i->ij", standard_Wilson_operator._matrix_coordinates_space_component_of_operator_function((+1, 0)), (standard_Wilson_operator._gauge_links_field_array).take(indices=1, axis=-1).reshape(-1) ), np.identity(standard_Wilson_operator.fermion_dimensions) - constants.gamma_matrices[1]) + np.kron(np.einsum("ij,i->ij", standard_Wilson_operator._matrix_coordinates_space_component_of_operator_function((-1, 0)), np.conjugate(np.roll((standard_Wilson_operator._gauge_links_field_array).take(indices=1, axis=-1), axis=0, shift=+1).reshape(-1)) ), np.identity(standard_Wilson_operator.fermion_dimensions) + constants.gamma_matrices[1])

    replicated_output += -0.5*gauged_terms_sum

    assert np.all(tested_output == replicated_output)

#####################################
####### PROPERTY-BASED TESTING ######
#####################################

@pytest.mark.property_based_test
@given(st.data())
def test_transpose_of_coordinates_space_component_of_operator_(data):
    fundamental_parameters = data.draw( generate_fundamental_parameters(), "fundamental_parameters")

    lattice_size, lattice_dimensions = fundamental_parameters

    standard_Wilson_operator = construct_StandardWilsonOperator_object(fundamental_parameters)

    unit_direction_tuple = data.draw(generate_unit_direction_tuple(lattice_dimensions=lattice_dimensions), "unit_direction_tuple")

    tested_expression = standard_Wilson_operator._matrix_coordinates_space_component_of_operator_function(unit_direction_tuple)
    tested_expression = np.transpose(tested_expression)

    opposite_unit_direction_tuple = tuple([-1*unit_direction_tuple[index] for index in range(lattice_dimensions)])

    benchmark_expression = standard_Wilson_operator._matrix_coordinates_space_component_of_operator_function(opposite_unit_direction_tuple)

    assert np.all(tested_expression == benchmark_expression)


@pytest.mark.property_based_test
@given(fundamental_parameters = generate_fundamental_parameters())
def test_gamma5_hermiticity_of_renormalized_mass_term_of_operator_matrix(fundamental_parameters):
    standard_Wilson_operator = construct_StandardWilsonOperator_object(fundamental_parameters)

    matrix_renormalized_mass_term_of_operator_array = standard_Wilson_operator._matrix_renormalized_mass_term_of_operator_function()

    benchmark_expression = np.transpose(np.conjugate(matrix_renormalized_mass_term_of_operator_array))

    extended_gamma5_matrix = constants.matrix_extended_gamma5_function(matrix_renormalized_mass_term_of_operator_array)

    tested_expression = np.matmul(extended_gamma5_matrix, np.matmul(matrix_renormalized_mass_term_of_operator_array, extended_gamma5_matrix))
    
    assert np.all(tested_expression == benchmark_expression)


@pytest.mark.property_based_test
@given(fundamental_parameters = generate_fundamental_parameters())
def test_gamma5_hermiticity_of_standard_Wilson_operator_matrix(fundamental_parameters):
    standard_Wilson_operator = construct_StandardWilsonOperator_object(fundamental_parameters)

    matrix_standard_Wilson_operator_array = standard_Wilson_operator.matrix_standard_Wilson_operator_function()

    benchmark_expression = np.transpose(np.conjugate(matrix_standard_Wilson_operator_array))

    extended_gamma5_matrix = constants.matrix_extended_gamma5_function(matrix_standard_Wilson_operator_array)

    tested_expression = np.matmul(extended_gamma5_matrix, np.matmul(matrix_standard_Wilson_operator_array, extended_gamma5_matrix))
    
    assert np.all(tested_expression == benchmark_expression)


@pytest.mark.property_based_test
@given(st.data())
def test_gamma5_hermiticity_of_pairs_of_directed_gauged_terms_of_opposite_direction(data):
    fundamental_parameters = data.draw( generate_fundamental_parameters(), "fundamental_parameters")

    lattice_size, lattice_dimensions = fundamental_parameters

    standard_Wilson_operator = construct_StandardWilsonOperator_object(fundamental_parameters)

    unit_direction_tuple = data.draw(generate_unit_direction_tuple(lattice_dimensions=lattice_dimensions), "unit_direction_tuple")
    
    opposite_unit_direction_tuple = tuple([-1*unit_direction_tuple[index] for index in range(lattice_dimensions)])

    sum_of_directed_gauged_terms_of_opposite_direction = standard_Wilson_operator._matrix_directed_gauged_term_of_operator_function(unit_direction_tuple) + standard_Wilson_operator._matrix_directed_gauged_term_of_operator_function(opposite_unit_direction_tuple)

    benchmark_expression = np.transpose(np.conjugate(sum_of_directed_gauged_terms_of_opposite_direction))
    
    extended_gamma5_matrix = constants.matrix_extended_gamma5_function(sum_of_directed_gauged_terms_of_opposite_direction)

    tested_expression = np.matmul(extended_gamma5_matrix, np.matmul(sum_of_directed_gauged_terms_of_opposite_direction, extended_gamma5_matrix))
    
    assert np.all(tested_expression == benchmark_expression)


if __name__ == "__main__":

    unittest.main()
