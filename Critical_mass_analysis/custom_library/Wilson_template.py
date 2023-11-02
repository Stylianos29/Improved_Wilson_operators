'''
This module contains the functions and the classes for an undoubled Wilson-Dirac operator class that can function as a template for the construction of improved Wilson operators classes.
'''

import numpy as np
import itertools as it

# import custom modules
import custom_library.constants as constants
import custom_library.lattice as lattice
import custom_library.gauge as gauge


class StandardWilsonOperator(gauge.GaugeLinksField):
    '''Note: Only for testing purposes.'''

    def __init__(self, lattice_size=9, lattice_dimensions=2, fermion_dimensions=4, gauge_theory_label=1, effective_bare_mass_value=0., *, temporal_axis_size=None):
        '''This base class constructor is identical to the one of the alternative constructor of the "GaugeLinksField" class: .from_fundamental_parameters_with_random_gauge_links_field(...).'''

        super().__init__(random_gauge_links_field_boolean=True, gauge_theory_label=gauge_theory_label, lattice_structure=lattice.LatticeStructure(lattice_size=lattice_size, lattice_dimensions=lattice_dimensions, fermion_dimensions=fermion_dimensions, temporal_axis_size=temporal_axis_size))

        self.effective_bare_mass_value = effective_bare_mass_value

    @property
    def effective_bare_mass_value(self):
        return self._effective_bare_mass_value
    
    @effective_bare_mass_value.setter
    def effective_bare_mass_value(self, effective_bare_mass_value):
        assert isinstance(effective_bare_mass_value, float) and (effective_bare_mass_value >= 0.), 'Dimensionless parameter "effective_bare_mass_value" must be a positive real number.'

        self._effective_bare_mass_value = effective_bare_mass_value

    def __repr__(self) -> str:
        return f"\n{type(self).__name__}(lattice_size={self._lattice_size}, lattice_dimensions={self._lattice_dimensions}, fermion_dimensions={self._fermion_dimensions}, gauge_theory_label={self._gauge_theory_label}, effective_bare_mass_value={self._effective_bare_mass_value:.3}, temporal_axis_size={self._lattice_shape[0]})\n"
    
    def __str__(self) -> str:
        '''The str output of the base class is enhanced with additional details about the effective mass value of the fermion field.'''
        
        str_output = super().__str__()

        str_output += f'The effective bare mass value of the fermion field is: {self._effective_bare_mass_value}.\n'

        return str_output

    def matrix_standard_Wilson_operator_function(self, switching_of_gauge_field_boolean=False):
        ''''''
        
        # renormalized mass term
        matrix_standard_Wilson_operator_array = self._matrix_renormalized_mass_term_of_operator_function()

        # gauged terms
        matrix_sum_of_gauged_terms_array = np.zeros_like(matrix_standard_Wilson_operator_array, dtype=np.complex_)
        for unit_direction_tuple in self._list_of_unit_direction_tuples_function():
            matrix_sum_of_gauged_terms_array += self._matrix_directed_gauged_term_of_operator_function(unit_direction_tuple, switching_of_gauge_field_boolean=switching_of_gauge_field_boolean)
        
        matrix_standard_Wilson_operator_array += -0.5*matrix_sum_of_gauged_terms_array

        return matrix_standard_Wilson_operator_array

    def eigenvalues_spectrum_function(self):
        pass

    def _list_of_unit_direction_tuples_function(self):
        '''-OUTPUT: list of tuples of length "size".
        -NOTE: Performance check against other method.'''

        unit_directions_list = list()
        # Initial unit direction as list object that can be mutated
        unit_direction = [1]+[0]*(self._lattice_dimensions-1)
        for index in range(self._lattice_dimensions):
            # Append positive direction
            unit_directions_list.append(tuple(unit_direction.copy()))
            # Append negative direction
            unit_direction[index] = -1*unit_direction[index]
            unit_directions_list.append(tuple(unit_direction.copy()))
            # Construct next 
            unit_direction.insert(index+1,unit_direction.pop(index))

        return unit_directions_list 

    def _matrix_renormalized_mass_term_of_operator_function(self):
        '''-USAGE: Kronecker product of 3 matrices in this specific order: coordinate space identity matrix ⊗ gauge structure identity matrix ⊗ Dirac structure identity matrix.
        -OUTPUT: 2D square array of side length (L^d)*(u^2)*f.'''

        renormalized_mass_value = self._effective_bare_mass_value + self._lattice_dimensions

        matrix_coordinate_space_operator_array = np.identity(self._lattice_size**self._lattice_dimensions)
        directed_gauge_links_field_array = np.identity(self.gauge_theory_label)
        Dirac_gamma_matrix_array = np.identity(self.fermion_dimensions, dtype=np.complex_)
    
        return renormalized_mass_value*np.kron(matrix_coordinate_space_operator_array, np.kron(directed_gauge_links_field_array, Dirac_gamma_matrix_array))

    def _matrix_directed_gauged_term_of_operator_function(self, unit_direction_tuple, switching_of_gauge_field_boolean=False):
        '''-INPUT: tuple of size d containing only non-zero element with value -1 or +1.
        -USAGE: Kronecker product of 3 matrices: coordinate space delta function matrix ⊗ gauge structure matrix ⊗ Dirac structure matrix, all three depending on the input direction tuple passed.
        -OUTPUT: 2D square array of side length (L^d)*(u^2)*f '''

        matrix_coordinate_space_operator_array = self._matrix_coordinates_space_component_of_operator_function(unit_direction_tuple)

        matrix_directed_gauge_links_field_array = self._directed_gauge_links_field_function(unit_direction_tuple)
        
        if switching_of_gauge_field_boolean:
            matrix_directed_gauge_links_field_array = np.ones((self._lattice_size, self._lattice_size))

        '''Since a "unit_direction_tuple" contains only one non-zero element, with value with either +1 or -1, then summing this tuple will result to a value +1 or -1, indicating a positive or a negative direction correspondingly.'''
        gamma_matrix_sign_factor = sum(unit_direction_tuple)

        gamma_matrix_index = self._mapping_lattice_direction_to_axis_index(unit_direction_tuple)

        matrix_Dirac_structure_array = np.identity(self.fermion_dimensions) - gamma_matrix_sign_factor*constants.gamma_matrices[gamma_matrix_index]

        if (self._gauge_theory_label == 1):
            '''Every row of the coordinate space delta function matrix is element-wise multiplied by the flattened gauge structure matrix. And then the output is simply the Kronecker product of the resulting matrix with the Dirac structure matrix.'''

            matrix_gauged_coordinate_space_operator_array = np.einsum("ij,i->ij",matrix_coordinate_space_operator_array, matrix_directed_gauge_links_field_array.reshape(-1))

            return np.kron(matrix_gauged_coordinate_space_operator_array, matrix_Dirac_structure_array)
        
        else:
            '''????????Every row of the coordinate space delta function matrix is element-wise multiplied by the flattened gauge structure matrix. And then the output is simply the Kronecker product of the resulting matrix with the Dirac structure matrix.'''

            reshape_tuple = (self._lattice_size**self._lattice_dimensions, self._gauge_theory_label, self._gauge_theory_label)

            matrix_gauged_coordinate_space_operator_array = np.einsum("ij,j...->ij...",matrix_coordinate_space_operator_array, matrix_directed_gauge_links_field_array.reshape(reshape_tuple))

            blocked_operator_matrix_directed_gauged_term = [[np.kron(matrix_gauged_coordinate_space_operator_array[row, column], matrix_Dirac_structure_array) for column in range(self._lattice_size**self._lattice_dimensions)] for row in range(self._lattice_size**self._lattice_dimensions)]

            return np.block(blocked_operator_matrix_directed_gauged_term)

    def _mapping_lattice_direction_to_axis_index(self, unit_direction_tuple):
        '''Iterating over the elements of "unit_direction_tuple", stopping at the first non-zero element, and passing its index as the output.'''

        index = next((i for i, x in enumerate(unit_direction_tuple) if x), None)

        return self._lattice_dimensions - 1 - index

    def _matrix_coordinates_space_component_of_operator_function(self, unit_direction_tuple):
        '''-INPUT:
        -USAGE:
        -OUTPUT:
        -NOTE:'''

        coordinates_space_operator_side_length = self._lattice_size**self._lattice_dimensions

        matrix_coordinate_space_kronecker_delta_function_array = np.zeros((coordinates_space_operator_side_length, coordinates_space_operator_side_length), dtype=np.complex_)

        for n in it.product(range(self._lattice_size), repeat=self._lattice_dimensions):
            m = self.lattice_sites_coordinates_addition(n, unit_direction_tuple)

            x = self._mapping_lattice_sites_coordinates_to_operator_matrix_coordinate(n)
            y = self._mapping_lattice_sites_coordinates_to_operator_matrix_coordinate(m)
            matrix_coordinate_space_kronecker_delta_function_array[x, y] = 1.

        return matrix_coordinate_space_kronecker_delta_function_array

    def _mapping_lattice_sites_coordinates_to_operator_matrix_coordinate(self, lattice_site_coordinates_tuple):
        '''-INPUT: d-sized tuple representing the coordinates of a lattice site.
        -USAGE: Mapping a lattice site coordinate to either a row or a column index of the 2D matrix coordinate space structure of the Dirac operator.
        -OUTPUT: a single (positive integer) number
        -NOTE:'''

        operator_coordinate_space_structure_coordinate = 0
        for index in range(self._lattice_dimensions):
            operator_coordinate_space_structure_coordinate += lattice_site_coordinates_tuple[index]*self._lattice_size**(self._lattice_dimensions-index-1)

        return operator_coordinate_space_structure_coordinate%(self._lattice_size**self._lattice_dimensions)

    def _directed_gauge_links_field_function(self, unit_direction_tuple):
        
        gauge_links_field_array_direction = self._mapping_lattice_direction_to_axis_index(unit_direction_tuple)

        if (self._gauge_theory_label == 1):
            directed_gauge_links_field_array = (self._gauge_links_field_array).take(indices=gauge_links_field_array_direction, axis=-1)

        else:
            directed_gauge_links_field_array = (self._gauge_links_field_array).take(indices=gauge_links_field_array_direction, axis=-3)

        if (sum(unit_direction_tuple) < 0):
            directed_gauge_links_field_array = np.roll(directed_gauge_links_field_array, axis= self.lattice_dimensions - 1 - (gauge_links_field_array_direction), shift=+1)

            if (self._gauge_theory_label == 1):
                directed_gauge_links_field_array = np.conjugate(directed_gauge_links_field_array)
            else:
                directed_gauge_links_field_array = np.conjugate(np.moveaxis(directed_gauge_links_field_array, -1, -2))

        return directed_gauge_links_field_array
