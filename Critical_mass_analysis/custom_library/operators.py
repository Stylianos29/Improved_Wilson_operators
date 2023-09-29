'''Definitions of classes representing the improved Wilson operators and overlap operators.'''
import numpy as np
from mpi4py import MPI
from scipy.sparse.linalg import LinearOperator

# import custom libraries
from custom_library.gauge import GaugeLinksField
import custom_library.constants as constants
import custom_library.Chebyshev_polynomials as Chebyshev_polynomials
import custom_library.Kenney_Laub_iterates as Kenney_Laub_iterates


class BaseDiracOperator:

    def __init__():
        pass
    
    def _reproducing_sparse_operator_matrix(self, Dirac_operator_linear_function, size, number_of_columns=4):
        ''''''

        psi_vector_size = size

        matrix_Dirac_operator_2D_array = map(
            self.Dirac_operator_linear_function, np.eye(number_of_columns, psi_vector_size, dtype=np.complex_)
        )
        matrix_Dirac_operator_2D_array = np.array(list(matrix_Dirac_operator_2D_array))

        return matrix_Dirac_operator_2D_array.T
    
    '''
    TODO: Bring gamma5_function(self, x) here
    '''

    '''
    TODO: Create an inverse function here
    '''
    
class ImprovedWilsonOperator(GaugeLinksField):

    def __init__(self, gauge_links_field, derivative_stencil_index=0, laplacian_stencil_index=0, bare_mass_value=0.):

        assert isinstance(gauge_links_field, GaugeLinksField), 'The argument must be an instance of the "GaugeLinksField" class.'
        super().__init__(gauge_links_field.lattice_size, gauge_links_field.lattice_dimensions, gauge_links_field.fermion_dimensions, gauge_links_field.theory_label)

        assert not (hasattr(gauge_links_field, "gauge_links_field.gauge_links_phase_values_field_array") == AttributeError), 'The passed instance of the GaugeLinksField class must contain an array with the gauge links field phase values.'
        self.gauge_links_phase_values_field_array = gauge_links_field._gauge_links_phase_values_field_array

        assert isinstance(derivative_stencil_index, int) and derivative_stencil_index in np.arange(np.shape(constants.derivative_stencils_array)[1]), f'The index choice corresponding to the derivative stencil must be in the range {np.arange(np.shape(constants.derivative_stencils_array)[1])}.'
        self.derivative_stencil_index = derivative_stencil_index
        self.derivative_stencil = list()
        for coordinate_direction in range(self.lattice_dimensions):
            self.derivative_stencil.append(constants.derivative_stencils_array[coordinate_direction][derivative_stencil_index])

        assert isinstance(laplacian_stencil_index, int) and laplacian_stencil_index in np.arange(len(constants.laplacian_stencils_array)), f'The index choice corresponding to the derivative stencil must be in the range {np.arange(len(constants.laplacian_stencils_array))}.'
        self.laplacian_stencil_index = laplacian_stencil_index
        # Include the mass term in the laplacian stencil for convenience
        '''
        TODO: Decide whether to distinguish the mass term from the laplacian stencil or not.
        '''
        mass_term_array = np.array([[0, 0, 0], [0, -bare_mass_value, 0], [0, 0, 0]])
        self.laplacian_stencil = constants.laplacian_stencils_array[laplacian_stencil_index] + mass_term_array
        assert isinstance(bare_mass_value, float) and bare_mass_value >= 0., 'The fermion field bare mass value must be a non-negative real number.'
        self.bare_mass_value = bare_mass_value
    
    @classmethod
    def from_fundamental_parameters(cls, lattice_size, lattice_dimensions, fermion_dimensions, theory_label, derivative_stencil_index, laplacian_stencil_index, bare_mass_value, random_gauge_links_field_boolean=False):
        '''For testing purposes only.'''

        return cls(GaugeLinksField(lattice_size, lattice_dimensions, fermion_dimensions, theory_label, random_gauge_links_field_boolean), derivative_stencil_index, laplacian_stencil_index, bare_mass_value)

    def __repr__(self) -> str:
        '''
        TODO: Rewrite it: refer to the lattice, the gauge field and then the Dirac operator.
        '''

        return f'\nAn instance of the {type(self).__name__} class has been created with arguments: lattice_size={self.lattice_size}, lattice_dimensions={self.lattice_dimensions}, fermion_dimensions={self.fermion_dimensions}, theory_label={self.theory_label}, derivative_stencil_index={self.y_direction_derivative_stencil_index}, laplacian_stencil_index={self.laplacian_stencil_index}.\n'

    def improved_Wilson_operator_linear_function(self, psi_vector_field):
        '''An improved Wilson operator is composed of a derivative part, with one term per coordinate direction, and a laplacian part.'''

        assert np.product(np.shape(psi_vector_field)) == self.fermion_dimensions*self.lattice_size**self.lattice_dimensions*self.theory_label**2, 'Input must be a column vector of size (f*L^d*u^2,).'

        # Calculate each term of the derivative part in coordinate space as (L, L, f, u, u) column vector
        coordinate_space_derivative_columns = list()
        for coordinate_direction in range(self.lattice_dimensions):
            coordinate_space_derivative_columns.append(self._coordinate_space_improved_Wilson_operator_linear_function(self.derivative_stencil[coordinate_direction], psi_vector_field))

        # Calculate the laplacian part in coordinate space as (L, L, f, u, u) column vector
        coordinate_space_laplacian_column = self._coordinate_space_improved_Wilson_operator_linear_function(self.laplacian_stencil, psi_vector_field)

        # Dot product
        improved_Wilson_operator_column = np.zeros(np.shape(coordinate_space_laplacian_column), dtype=np.complex_)
        for coordinate_direction in range(self.lattice_dimensions):
            improved_Wilson_operator_column += np.einsum("...j,ij->...i", coordinate_space_derivative_columns[coordinate_direction], constants.gamma_matrices[coordinate_direction])
        improved_Wilson_operator_column += - 0.5*np.einsum("...j,ij->...i", coordinate_space_laplacian_column, np.identity(self.fermion_dimensions, dtype=np.complex_))

        return improved_Wilson_operator_column.reshape(-1)

    def _coordinate_space_improved_Wilson_operator_linear_function(self, stencil, psi_vector_field):

        assert np.product(np.shape(psi_vector_field)) == self.fermion_dimensions*self.lattice_size**self.lattice_dimensions*self.theory_label**2, 'Input must be a column vector of size (f*L^d*u^2,).'

        # Input vector needs to be reshape to an (L, L, f, u, u) shape
        reshaped_psi_vector_field_shape = (*[self.lattice_size]*self.lattice_dimensions, self.fermion_dimensions)
        # For SU(2) and SU(3) gauge theories it need additional 2 trailing components
        if (self.theory_label != 1):
            reshaped_psi_vector_field_shape = (*reshaped_psi_vector_field_shape, self.theory_label, self.theory_label)
        psi_vector_field = psi_vector_field.reshape(reshaped_psi_vector_field_shape)

        operator_column = np.zeros(np.shape(psi_vector_field), dtype=np.complex_)
        for stencil_row in [0, 1, 2]:
            for stencil_column in [0, 1, 2]:
                factor = stencil[stencil_row, stencil_column]
                if (factor != 0):
                    direction = (stencil_row-1, stencil_column-1)
                    gauge_links_field_array = self.gauge_links_field_function(self.gauge_links_phase_values_field_array, direction)
                    shifted_psi_vector_field = np.roll(np.roll(psi_vector_field, axis=0, shift=-direction[0]), axis=1, shift=-direction[1])
                    gauged_psi_vector_field = np.einsum("ij,ij...->ij...", gauge_links_field_array, shifted_psi_vector_field)
                    operator_column += factor*gauged_psi_vector_field

        return operator_column

    def _reproducing_sparse_operator_matrix(self, number_of_columns=4):
        ''''''

        psi_vector_size = self.fermion_dimensions*self.lattice_size**self.lattice_dimensions*self.theory_label**2

        matrix_improved_Wilson_operator = map(
            self.improved_Wilson_operator_linear_function, np.eye(number_of_columns, psi_vector_size, dtype=np.complex_)
        )
        matrix_improved_Wilson_operator = np.array(list(matrix_improved_Wilson_operator))

        return matrix_improved_Wilson_operator.T
    
    def gamma5_function(self, x):

        '''
        TODO: check input
        '''
        
        vector_size = np.shape(x)[0]

        reshaped_vector = x.reshape(vector_size//self.fermion_dimensions, self.fermion_dimensions)

        result = np.einsum("ij,...j->...i", constants.gamma_matrices[4], reshaped_vector)

        return result.reshape(-1)

    def inverse_Wilson_operator_function(self, precision=1E-4, number_of_rows=4):

        x0 = np.zeros((self.fermion_dimensions*(self.lattice_size**self.lattice_dimensions),), dtype=np.complex_)
        b = np.zeros((self.fermion_dimensions*(self.lattice_size**self.lattice_dimensions),), dtype=np.complex_)
        b[0] = 1.0

        inverse_improved_Wilson_operator = []
        for next_permutation in range(number_of_rows):
            
            inverse_improved_Wilson_operator_vector = self.improved_Wilson_operator_functional_conjugate_gradient_solution_function(b, x0, precision)
            
            inverse_improved_Wilson_operator.append(inverse_improved_Wilson_operator_vector)

            b = np.roll(b, shift=1)

        inverse_improved_Wilson_operator = (np.array(inverse_improved_Wilson_operator))

        return inverse_improved_Wilson_operator.T

    def improved_Wilson_operator_functional_conjugate_gradient_solution_function(self, b, x0, precision=1E-4):

        # helper function
        def improved_Wilson_operator_squared(x):
            return self.gamma5_function(self.improved_Wilson_operator_linear_function(self.gamma5_function(self.improved_Wilson_operator_linear_function(x))))
        

        b_prime = self.gamma5_function(self.improved_Wilson_operator_linear_function(self.gamma5_function(b)))
        b_norm = np.linalg.norm(b_prime)

        r = b_prime - improved_Wilson_operator_squared(x0)
        residue_norm = np.linalg.norm(r)

        p = r
        x = x0

        counter = 0
        while True:

            residue_norm = np.linalg.norm(r)

            Adotp = improved_Wilson_operator_squared(p)

            denominator = np.absolute((np.conjugate(p)).dot(Adotp))
            
            alpha_factor = (residue_norm)**2/ denominator

            x = x + alpha_factor*p
            r = r - alpha_factor*Adotp

            if (np.linalg.norm(r)/b_norm < precision):
                break

            beta_factor = (np.linalg.norm(r) / residue_norm)**2
            p = r + beta_factor*p

            counter += 1
        
        return x
    
    def time_dependent_pion_correlator_function(self, partial_inverse_Wilson_operator):
        # partial_inverse_Wilson_operator = inverse_Wilson_operator[:,:4]
        reshaped_inverse_Wilson_operator = partial_inverse_Wilson_operator.reshape(self.lattice_size**self.lattice_dimensions, self.fermion_dimensions, self.fermion_dimensions)
        squared_reshaped_inverse_Wilson_operator = np.einsum("...ij,...kj->...ik", reshaped_inverse_Wilson_operator, np.conjugate(reshaped_inverse_Wilson_operator))
        traced_squared_reshaped_inverse_Wilson_operator = np.einsum("...ii", squared_reshaped_inverse_Wilson_operator)
        reshaped_traced_squared_reshaped_inverse_Wilson_operator = traced_squared_reshaped_inverse_Wilson_operator.reshape(self.lattice_size, self.lattice_size)
        time_dependent_pion_correlator = np.sum(reshaped_traced_squared_reshaped_inverse_Wilson_operator, axis=0)

        return time_dependent_pion_correlator



    # MATRICES
    
    def _improved_Wilson_operator_sparse_matrix_function(self):

        coordinate_space_x_derivative_array = self._coordinate_space_matrix_generation_function(self.gauge_links_phase_values_field_array, self.y_direction_derivative_stencil)

        coordinate_space_y_derivative_array = self._coordinate_space_matrix_generation_function(self.gauge_links_phase_values_field_array, self.y_direction_derivative_stencil.T)

        coordinate_space_laplacian_array = self._coordinate_space_matrix_generation_function(self.gauge_links_phase_values_field_array, self.laplacian_stencil)

        improved_Wilson_operator_matrix = np.kron(coordinate_space_x_derivative_array, constants.gamma_matrices[0]) + np.kron(coordinate_space_y_derivative_array, constants.gamma_matrices[1]) - 0.5*np.kron(coordinate_space_laplacian_array, np.identity(self.fermion_dimensions, dtype=np.complex_))

        return improved_Wilson_operator_matrix

    def _sparse_operator_matrix_generation_function(self, direction, gauge_links_field):
        ''' Input: direction is a 2D tuple and gauge_links_field is L by L NumPy 2D array'''
        
        matrix_dimension = self.lattice_size**self.lattice_dimensions
        sparse_matrix = np.zeros((matrix_dimension, matrix_dimension), dtype=np.complex_)
        lattice_sites_coordinates_array = self.lattice_sites_coordinates(self.lattice_size, self.lattice_dimensions)

        for row_coordinate in lattice_sites_coordinates_array:
            for column_coordinate in lattice_sites_coordinates_array:
                if (self.tuple_addition(row_coordinate, direction) == column_coordinate):
                    sparse_matrix[self.coordinate_index(row_coordinate), self.coordinate_index(column_coordinate)] = gauge_links_field[row_coordinate]

        return sparse_matrix

    def _coordinate_space_matrix_generation_function(self, stencil):

        Wilson_operator_dimension = self.lattice_size**self.lattice_dimensions
        gauged_coordinate_space_matrix = np.zeros((Wilson_operator_dimension, Wilson_operator_dimension), dtype=np.complex_)
        for stencil_row in [0, 1, 2]:
            for stencil_column in [0, 1, 2]:
                factor = stencil[stencil_row, stencil_column]
                if (factor != 0):
                    direction = (stencil_row-1, stencil_column-1)
                    if direction==(-1,0):
                        break
                    gauge_links_field_array = self.gauge_links_field_function(self.gauge_links_phase_values_field_array, direction)
                    gauged_coordinate_space_matrix += factor*self._sparse_operator_matrix_generation_function(direction, gauge_links_field_array)

        return gauged_coordinate_space_matrix
    
    def _matrix_hard_coded_standard_Wilson_operator_function(self):
        '''Coordinate structure kronecker product gauge structure kronecker product Dirac structure.'''

        def projected_gauge_links_field(direction_index):
            return ((self.gauge_links_phase_values_field_array.T)[:,:,direction_index]).T

        temp = np.arange(-self.lattice_dimensions, self.lattice_dimensions+1)
        temp = temp[temp != 0]

        for direction_index in temp:
            
            coordinate_structure = np.identity(self.lattice_size**self.lattice_dimensions, dtype=np.complex_)

            gauge_structure = projected_gauge_links_field(abs(direction_index)-1)
            gauge_structure = gauge_structure.reshape(self.lattice_size**self.lattice_dimensions, self.theory_label, self.theory_label)

            block_arrays = np.einsum("i...,i...->i...", coordinate_structure[:, :, np.newaxis, np.newaxis], gauge_structure)

            kronecker_product = np.block([[block_arrays[row, columns] for columns in range(self.lattice_size**self.lattice_dimensions)] for row in range(self.lattice_size**self.lattice_dimensions)])

        return (self.bare_mass_value + self.lattice_dimensions)*np.kron(np.identity(self.lattice_size**self.lattice_dimensions, dtype=np.complex_), np.kron(np.identity(self.theory_label, dtype=np.complex_), np.identity(self.fermion_dimensions, dtype=np.complex_))) - 0.5*np.kron(kronecker_product, np.identity(4, dtype=np.complex_) - np.sign(direction_index)*constants.gamma_matrices[abs(direction_index)-1])

############################################################
############################################################
############################################################
        
    def parallel_improved_Wilson_operator_column_function(self, stencil, partial_psi_vector_field):

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        stencil = np.array(stencil)

        partial_psi_vector_field = partial_psi_vector_field.reshape(self.lattice_size//size+2, self.lattice_size, self.fermion_dimensions)

        operator_column = np.zeros(np.shape(partial_psi_vector_field), dtype=np.complex_)
        for stencil_row in [0, 1, 2]:
            for stencil_column in [0, 1, 2]:
                factor = stencil[stencil_row, stencil_column]
                if (factor != 0.):
                    direction = (stencil_row-1, stencil_column-1)
                    gauge_links_field_array = self.gauge_links_field_function(self._gauge_links_phase_values_field_array, direction)
                    shifted_psi_vector_field = np.roll(np.roll(partial_psi_vector_field, axis=0, shift=-direction[0]), axis=1, shift=-direction[1])
                    gauged_psi_vector_field = np.einsum("ij,ij...->ij...", gauge_links_field_array, shifted_psi_vector_field)
                    operator_column += factor*gauged_psi_vector_field
        
        # Discard the halo
        # operator_column = operator_column[1:self.lattice_size//size+1]

        return operator_column

    def parallel_improved_Wilson_operator_linear_function(self, psi_vector_field):

        spatial_x_derivative_column = self.parallel_improved_Wilson_operator_column_function(self.derivative_stencil[0], psi_vector_field)

        spatial_y_derivative_column = self.parallel_improved_Wilson_operator_column_function(self.derivative_stencil[1], psi_vector_field)

        spatial_laplacian_column = self.parallel_improved_Wilson_operator_column_function(self.laplacian_stencil, psi_vector_field)


        improved_Wilson_operator_column = np.einsum("...j,ij->...i", spatial_x_derivative_column, constants.gamma_matrices[0]) + np.einsum("...j,ij->...i", spatial_y_derivative_column, constants.gamma_matrices[1]) - 0.5*np.einsum("...j,ij->...i", spatial_laplacian_column, np.identity(self.fermion_dimensions, dtype=np.complex_))

        return improved_Wilson_operator_column.reshape(-1)

    def parallel_inverse_Wilson_operator_function(self, precision=1E-4, number_of_rows=4):

            x0 = np.zeros((self.fermion_dimensions*(self.lattice_size**self.lattice_dimensions),), dtype=np.complex_)
            b = np.zeros((self.fermion_dimensions*(self.lattice_size**self.lattice_dimensions),), dtype=np.complex_)
            b[0] = 1.0

            inverse_improved_Wilson_operator = []
            for next_permutation in range(number_of_rows):
                
                inverse_improved_Wilson_operator_vector = self.improved_Wilson_operator_functional_conjugate_gradient_solution_function(b, x0, precision)
                
                inverse_improved_Wilson_operator.append(inverse_improved_Wilson_operator_vector)

                b = np.roll(b, shift=1)

            inverse_improved_Wilson_operator = (np.array(inverse_improved_Wilson_operator))

            return inverse_improved_Wilson_operator.T

    def parallel_improved_Wilson_operator_functional_conjugate_gradient_solution_function(self, b, x0, precision=1E-4):

        # helper functions
        def reconfigure_local_vector(local_vector):
            '''Reconfigure the halo above and below the extended local vector.'''

            L = self.lattice_size
            d = self.lattice_dimensions
            f = self.fermion_dimensions
        
            # Append a halo to the initial vector field
            reconfigured_local_vector = local_vector
            # Lower element addition
            self.comm.Sendrecv(local_vector[f*(L//self.size)*L:f*(L//self.size+1)*L], dest=(self.rank+1)%self.size, sendtag=(self.rank+1)%self.size, recvbuf=reconfigured_local_vector[:f*L], source=(self.rank-1)%self.size, recvtag=self.rank)
            # Upper element addition
            self.comm.Sendrecv(np.array(local_vector[f*L:2*f*L]), dest=(self.rank-1)%self.size, sendtag=(self.rank-1)%self.size, recvbuf=reconfigured_local_vector[f*(L//self.size+1)*L:], source=(self.rank+1)%self.size, recvtag=self.rank)

            return reconfigured_local_vector
        
        def improved_Wilson_operator_squared(x):
            
            projected_column = self.gamma5_function(self.parallel_improved_Wilson_operator_linear_function(x))
            
            reconfigured_projected_column = reconfigure_local_vector(projected_column)
            
            return self.gamma5_function(self.parallel_improved_Wilson_operator_linear_function(reconfigured_projected_column))

        def collective_norm(local_vector):
            # Calculate the collective norm of residue_norm
            truncated_local_vector = local_vector[self.fermion_dimensions*self.lattice_size:self.fermion_dimensions*(self.lattice_size//self.size+1)*self.lattice_size]
            local_sum_of_squared_terms = np.sum(truncated_local_vector*np.conjugate(truncated_local_vector))
            local_vector_norm_squared = np.zeros_like(local_sum_of_squared_terms)
            self.comm.Allreduce(local_sum_of_squared_terms, local_vector_norm_squared, op = MPI.SUM)

            return np.sqrt(np.abs(local_vector_norm_squared))
        

        b_prime = self.gamma5_function(self.parallel_improved_Wilson_operator_linear_function(self.gamma5_function(b)))
        b_norm = collective_norm(b_prime)

        r = b_prime - improved_Wilson_operator_squared(x0)
        residue_norm = collective_norm(r)

        p = r
        x = x0

        counter = 0
        while True:

            residue_norm = collective_norm(r)

            Adotp = improved_Wilson_operator_squared(p)

            p_prime = self.gamma5_function(self.parallel_improved_Wilson_operator_linear_function(p))

            denominator = collective_norm(p_prime)

            alpha_factor = (residue_norm)**2 / denominator**2

            x = x + alpha_factor*p
            r = r - alpha_factor*Adotp

            new_residue_norm = collective_norm(r)

            if (new_residue_norm/b_norm < precision):
                break

            beta_factor = (new_residue_norm / residue_norm)**2
            p = r + beta_factor*p

            counter += 1
        
        return x

    # def parallel_improved_Wilson_operator_linear_function(self, partial_gauge_links_phase_values_field, derivative_stencil, laplacian_stencil, lattice_size, partial_psi_vector_field):

    #     comm = MPI.COMM_WORLD
    #     rank = comm.Get_rank()
    #     size = comm.Get_size()

    #     L = lattice_size

    #     extended_partial_gauge_links_phase_values_field = np.zeros((L//size+2, L, 2))
    #     extended_partial_gauge_links_phase_values_field[1:L//size+1,:] = partial_gauge_links_phase_values_field

    #     extended_partial_psi_vector_field = np.zeros((4*(L//size+2)*L,), dtype=np.complex_)
    #     extended_partial_psi_vector_field[4*L:4*(L//size+1)*L] = partial_psi_vector_field

    #     # Upper addition
    #     comm.Sendrecv(partial_psi_vector_field[4*(L//size-1)*L:], dest=(rank+1)%size, sendtag=(rank+1)%size, recvbuf=extended_partial_psi_vector_field[:4*L], source=(rank-1)%size, recvtag=rank)
    #     # Lower addition
    #     comm.Sendrecv(np.array(partial_psi_vector_field[:4*L]), dest=(rank-1)%size, sendtag=(rank-1)%size, recvbuf=extended_partial_psi_vector_field[4*(L//size+1)*L:], source=(rank+1)%size, recvtag=rank)

    #     # Upper halo
    #     comm.Sendrecv(local_data[-1,:], dest=(rank+1)%size, sendtag=(rank+1)%size, recvbuf=extended_partial_gauge_links_phase_values_field[0,:], source=(rank-1)%size, recvtag=rank)

    #     # Lower halo
    #     comm.Sendrecv(local_data[0,:], dest=(rank-1)%size, sendtag=(rank-1)%size, recvbuf=extended_partial_gauge_links_phase_values_field[-1,:], source=(rank+1)%size, recvtag=rank)

    #     partial_improved_Wilson_operator_vector = self.partial_improved_Wilson_operator_linear_function(extended_partial_psi_vector_field)

    #     return partial_improved_Wilson_operator_vector



############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################


class OverlapOperator(ImprovedWilsonOperator):
    
    def __init__(self, improved_Wilson_operator, rho=1.):
        '''
        TODO: include in the arguments the module with the mathematical functions
        '''

        assert isinstance(improved_Wilson_operator, ImprovedWilsonOperator), 'The argument must be an instance of the "ImprovedWilsonOperator" class.'
        gauge_links_field = GaugeLinksField(improved_Wilson_operator.lattice_size, improved_Wilson_operator.lattice_dimensions, improved_Wilson_operator.fermion_dimensions, improved_Wilson_operator.theory_label)
        
        gauge_links_field.gauge_links_phase_values_field_array = ImprovedWilsonOperator.gauge_links_phase_values_field_array

        super().__init__(gauge_links_field, improved_Wilson_operator.derivative_stencil_index, improved_Wilson_operator.laplacian_stencil_index, improved_Wilson_operator.bare_mass_value)

        self.gauge_links_phase_values_field_array = improved_Wilson_operator._gauge_links_phase_values_field_array

        '''
        TODO: rho is a mutable parameter, it must be
        '''
        assert isinstance(rho, float) and rho > 0., 'Parameter rho value must be a positive real number.'
        self.rho = rho

        ''''
        TODO: Include and check choice of an external sign function.
        '''

    @classmethod
    def from_fundamental_parameters(cls, lattice_size, lattice_dimensions, fermion_dimensions, theory_label, derivative_stencil_index, laplacian_stencil_index, bare_mass_value, rho, sign_functions_module):
        '''For testing purposes only.'''

        return cls(ImprovedWilsonOperator._from_fundamental_parameters(lattice_size, lattice_dimensions, fermion_dimensions, theory_label, derivative_stencil_index, laplacian_stencil_index, bare_mass_value), rho, sign_functions_module)
    
    @classmethod
    def from_GaugeLinksField_object(cls, gauge_links_field, derivative_stencil_index, laplacian_stencil_index, bare_mass_value, rho=1.):

        return cls(ImprovedWilsonOperator(gauge_links_field, derivative_stencil_index, laplacian_stencil_index, bare_mass_value), rho)

    def __repr__(self) -> str:
        '''
        TODO: Rewrite it!
        '''

        return f'\nAn instance of the {type(self).__name__} class has been created with arguments: lattice_size={self.lattice_size}, lattice_dimensions={self.lattice_dimensions}, fermion_dimensions={self.fermion_dimensions}, theory_label={self.theory_label}, derivative_stencil_index={self.y_direction_derivative_stencil_index}, laplacian_stencil_index={self.laplacian_stencil_index}.\n'
    
    def massive_overlap_operator_linear_function(self, alpha, beta, x, N=50):
        return (1-0.5*self.bare_mass_value/self.rho)*self.overlap_operator_linear_function(alpha, beta, x, N) + self.bare_mass_value*x
    
    def overlap_operator_linear_function(self, alpha, beta, x, N=50):
        return x + self.gamma5_function(self.approximate_matrix_sign_column_function(alpha, beta, x, N))
    
    def approximate_matrix_sign_column_function(self, alpha, beta, x, N=50):

        def h_vector(x):
            return self.gamma5_function(self.improved_Wilson_operator_linear_function(x)) - self.gamma5_function(x)

        def X_column(x):
            return (1/(beta**2 - alpha**2))*( 2*h_vector(h_vector(x)) - (beta**2 + alpha**2)*x )

        def recursive_Chebyshev_polynomials_vector_function(n, x):
            if (n == 0):
                return x
            elif (n == 1):
                return X_column(x)
            else:
                return 2.0*X_column(recursive_Chebyshev_polynomials_vector_function(n-1, x)) - recursive_Chebyshev_polynomials_vector_function(n-2, x)

        def Chebyshev_polynomials_vector_function(n, x):
            Tn_minus1 = x
            if (n==0):
                return Tn_minus1
            else:
                Tn = X_column(x)
                for order in range(1, n):
                    temp = Tn
                    Tn = 2*X_column(Tn) - Tn_minus1
                    Tn_minus1 = temp

            return Tn

        vector_sum = np.zeros( (np.shape(x)[0]), dtype=np.complex_)
        for n in range(N):
            vector_sum += Chebyshev_polynomials.Chebyshev_polynomials_factors(n, N, alpha, beta)*Chebyshev_polynomials_vector_function(n, x)

        result = h_vector(vector_sum)

        return result

    
    def parallel_massive_overlap_operator_linear_function(self, alpha, beta, x, N=50):
        return (1-0.5*self.bare_mass_value/self.rho)*self.parallel_overlap_operator_linear_function(alpha, beta, x, N) + self.bare_mass_value*x
    
    def parallel_overlap_operator_linear_function(self, alpha, beta, x, N=50):
        return x + self.gamma5_function(self.parallel_approximate_matrix_sign_column_function(alpha, beta, x, N))
    
    def parallel_approximate_matrix_sign_column_function(self, alpha, beta, x, N=50):

        # helper functions
        def reconfigure_local_vector(local_vector):
            return local_vector
            '''Reconfigure the halo above and below the extended local vector.'''

            L = self.lattice_size
            d = self.lattice_dimensions
            f = self.fermion_dimensions
        
            # Append a halo to the initial vector field
            reconfigured_local_vector = local_vector
            # Lower element addition
            self.comm.Sendrecv(local_vector[f*(L//self.size)*L:f*(L//self.size+1)*L], dest=(self.rank+1)%self.size, sendtag=(self.rank+1)%self.size, recvbuf=reconfigured_local_vector[:f*L], source=(self.rank-1)%self.size, recvtag=self.rank)
            # Upper element addition
            self.comm.Sendrecv(np.array(local_vector[f*L:2*f*L]), dest=(self.rank-1)%self.size, sendtag=(self.rank-1)%self.size, recvbuf=reconfigured_local_vector[f*(L//self.size+1)*L:], source=(self.rank+1)%self.size, recvtag=self.rank)

            return reconfigured_local_vector


        def h_vector(x):
            return self.gamma5_function(self.parallel_improved_Wilson_operator_linear_function(x)) - self.gamma5_function(x)

        def X_column(x):
            temp = reconfigure_local_vector(h_vector(x))
            return (1/(beta**2 - alpha**2))*( 2*h_vector(temp) - (beta**2 + alpha**2)*x )

        def recursive_Chebyshev_polynomials_vector_function(n, x):
            if (n == 0):
                return x
            elif (n == 1):
                return X_column(x)
            else:
                return 2.0*X_column(recursive_Chebyshev_polynomials_vector_function(n-1, x)) - recursive_Chebyshev_polynomials_vector_function(n-2, x)

        def Chebyshev_polynomials_vector_function(n, x):
            Tn_minus1 = x
            if (n==0):
                return Tn_minus1
            else:
                Tn = X_column(x)
                for order in range(1, n):
                    Tn = reconfigure_local_vector(Tn)
                    temp = Tn
                    Tn = 2*X_column(Tn) - Tn_minus1
                    Tn_minus1 = temp

            return Tn

        vector_sum = np.zeros( (np.shape(x)[0]), dtype=np.complex_)
        for n in range(N):
            vector_sum += Chebyshev_polynomials.Chebyshev_polynomials_factors(n, N, alpha, beta)*Chebyshev_polynomials_vector_function(n, x)

        result = h_vector(vector_sum)

        return result


    def _reproducing_sparse_operator_matrix(self, number_of_columns=4):
        '''
        TODO: fix it!
        '''

        psi_vector_size = self.fermion_dimensions*self.lattice_size**self.lattice_dimensions*self.theory_label**2

        matrix_improved_Wilson_operator = map(
            self.massive_overlap_operator_linear_function, np.eye(number_of_columns, psi_vector_size, dtype=np.complex_)
        )
        matrix_improved_Wilson_operator = np.array(list(matrix_improved_Wilson_operator))

        return matrix_improved_Wilson_operator.T
    
    def inverse_overlap_operator_function(self, alpha, beta, number_of_rows=4):

        x0 = np.zeros((self.fermion_dimensions*(self.lattice_size**self.lattice_dimensions),), dtype=np.complex_)
        b = np.zeros((self.fermion_dimensions*(self.lattice_size**self.lattice_dimensions),), dtype=np.complex_)
        b[0] = 1.0

        inverse_overlap_operator = list()
        for next_permutation in range(number_of_rows):
            
            inverse_overlap_operator_vector = self.overlap_operator_functional_conjugate_gradient_solution_function(alpha, beta, b, x0)
            
            inverse_overlap_operator.append(inverse_overlap_operator_vector)

            b = np.roll(b, axis=0, shift=+1)

        inverse_overlap_operator = (np.array(inverse_overlap_operator))

        return inverse_overlap_operator.T
    
    def overlap_operator_functional_conjugate_gradient_solution_function(self, alpha, beta, b, x0, precision=1E-4, N=50):

        # helper functions
        def massive_overlap_operator(x):
            return self.massive_overlap_operator_linear_function(alpha, beta, x, N=N)

        def massive_overlap_operator_squared(x):
            return self.gamma5_function(massive_overlap_operator(self.gamma5_function(massive_overlap_operator(x))))

        b_prime = self.gamma5_function(massive_overlap_operator(self.gamma5_function(b)))
        b_norm = np.linalg.norm(b_prime)

        r = b_prime - massive_overlap_operator_squared(x0)
        residue_norm = np.linalg.norm(r)

        p = r
        x = x0

        # counter = 0
        while True:

            residue_norm = np.linalg.norm(r)

            Adotp = massive_overlap_operator_squared(p)

            denominator = np.absolute((np.conjugate(p)).dot(Adotp))

            alpha_factor = (residue_norm)**2/ denominator
            
            x = x + alpha_factor*p
            r = r - alpha_factor*Adotp

            if (np.linalg.norm(r)/b_norm < precision):
                break

            beta_factor = (np.linalg.norm(r) / residue_norm)**2
            p = r + beta_factor*p
        
            # counter += 1
        
        # print(counter)
        return x

    def parallel_overlap_operator_functional_conjugate_gradient_solution_function(self, alpha, beta, b, x0, precision=1E-4, N=50):

        # helper functions
        def reconfigure_local_vector(local_vector):
            '''Reconfigure the halo above and below the extended local vector.'''

            L = self.lattice_size
            d = self.lattice_dimensions
            f = self.fermion_dimensions
        
            # Append a halo to the initial vector field
            reconfigured_local_vector = local_vector
            # Lower element addition
            self.comm.Sendrecv(local_vector[f*(L//self.size)*L:f*(L//self.size+1)*L], dest=(self.rank+1)%self.size, sendtag=(self.rank+1)%self.size, recvbuf=reconfigured_local_vector[:f*L], source=(self.rank-1)%self.size, recvtag=self.rank)
            # Upper element addition
            self.comm.Sendrecv(np.array(local_vector[f*L:2*f*L]), dest=(self.rank-1)%self.size, sendtag=(self.rank-1)%self.size, recvbuf=reconfigured_local_vector[f*(L//self.size+1)*L:], source=(self.rank+1)%self.size, recvtag=self.rank)

            return reconfigured_local_vector
        
        def massive_overlap_operator(x):

            x = reconfigure_local_vector(x)

            return self.parallel_massive_overlap_operator_linear_function(alpha, beta, x, N=N)

        def massive_overlap_operator_squared(x):

            projected_column = self.gamma5_function(massive_overlap_operator(x))
            
            reconfigured_projected_column = reconfigure_local_vector(projected_column)
            
            return self.gamma5_function(massive_overlap_operator(reconfigured_projected_column))
        
        def collective_norm(local_vector):
            # Calculate the collective norm of residue_norm
            truncated_local_vector = local_vector[self.fermion_dimensions*self.lattice_size:self.fermion_dimensions*(self.lattice_size//self.size+1)*self.lattice_size]
            local_sum_of_squared_terms = np.sum(truncated_local_vector*np.conjugate(truncated_local_vector))
            local_vector_norm_squared = np.zeros_like(local_sum_of_squared_terms)
            self.comm.Allreduce(local_sum_of_squared_terms, local_vector_norm_squared, op = MPI.SUM)

            return np.sqrt(np.abs(local_vector_norm_squared))
        

        b_prime = self.gamma5_function(massive_overlap_operator(self.gamma5_function(b)))
        b_norm = collective_norm(b_prime)

        r = b_prime - massive_overlap_operator_squared(x0)
        residue_norm = collective_norm(r)

        p = r
        x = x0

        counter = 0
        while True:
            
            residue_norm = collective_norm(r)

            Adotp = massive_overlap_operator_squared(p)
        
            p_prime = self.gamma5_function(massive_overlap_operator(p))
        
            denominator = collective_norm(p_prime)

            alpha_factor = (residue_norm /  denominator )**2
            
            x = x + alpha_factor*p
            r = r - alpha_factor*Adotp

            new_residue_norm = collective_norm(r)

            if (new_residue_norm/b_norm < precision):
                break

            beta_factor = (new_residue_norm / residue_norm)**2
            p = r + beta_factor*p

            counter += 1
        
        return x
    
    ########################################################
    # KENNEY_LAUB_ITERATES
    def inverse_Kenney_Laub_iterate_overlap_operator_function(self, precision, number_of_rows=4):

            x0 = np.zeros((self.fermion_dimensions*(self.lattice_size**self.lattice_dimensions),), dtype=np.complex_)
            b = np.zeros((self.fermion_dimensions*(self.lattice_size**self.lattice_dimensions),), dtype=np.complex_)
            b[0] = 1.0

            inverse_Kenney_Laub_iterate_overlap_operator = list()
            for next_permutation in range(number_of_rows):
                
                inverse_Kenney_Laub_iterate_overlap_operator_vector = self.Kenney_Laub_iterate_overlap_operator_functional_conjugate_gradient_solution_function(b, x0, precision)
                
                inverse_Kenney_Laub_iterate_overlap_operator.append(inverse_Kenney_Laub_iterate_overlap_operator_vector)

                b = np.roll(b, axis=0, shift=+1)

            inverse_Kenney_Laub_iterate_overlap_operator = (np.array(inverse_Kenney_Laub_iterate_overlap_operator))

            return inverse_Kenney_Laub_iterate_overlap_operator.T
    
    def massive_Kenney_Laub_iterate_overlap_operator_linear_function(self, x, precision=1E-4):

        return (1-0.5*self.bare_mass_value)*self.Kenney_Laub_iterate_overlap_operator_linear_function(x, precision) + self.bare_mass_value*x
    
    def Kenney_Laub_iterate_overlap_operator_linear_function(self, x, precision=1E-4):

        return x + self.gamma5_function(self.approximate_matrix_Kenney_Laub_iterate_sign_function(x, precision))

    def approximate_matrix_Kenney_Laub_iterate_sign_function(self, x, precision=1E-4):
        # Sign function

        def h_vector(x):
            return self.gamma5_function(self.improved_Wilson_operator_linear_function(x)) - self.gamma5_function(x)
        
        x0 = np.zeros(np.shape(x), dtype=np.complex_)

        return (1./3.)*h_vector(x + (8./3.)*self.Kenney_Laub_iterate_denominator_inverse(x, x0, precision))

    def Kenney_Laub_iterate_denominator_inverse(self, b, x0, precision=1E-4):

        # helper functions
        def h_vector(x):
            return self.gamma5_function(self.improved_Wilson_operator_linear_function(x)) - self.gamma5_function(x)
        
        def Kenney_Laub_iterate_denominator_operator(x):
            return h_vector(h_vector(x)) + (1./3.)*x

        b_norm = np.linalg.norm(b)

        r = b
        # - Kenney_Laub_iterate_denominator_operator(x0)
        residue_norm = np.linalg.norm(r)

        p = r
        x = x0

        # counter = 0
        while True:

            residue_norm = np.linalg.norm(r)


            Adotp = Kenney_Laub_iterate_denominator_operator(p)

            denominator = np.absolute((np.conjugate(p)).dot(Adotp))

            alpha_factor = (residue_norm)**2/ denominator
            
            x = x + alpha_factor*p
            r = r - alpha_factor*Adotp

            if (np.linalg.norm(r)/b_norm < precision):
                break

            beta_factor = (np.linalg.norm(r) / residue_norm)**2
            p = r + beta_factor*p

            # counter += 1
        
        return x

    def Kenney_Laub_iterate_overlap_operator_functional_conjugate_gradient_solution_function(self, b, x0, precision=1E-4):

        # helper function
        def massive_Kenney_Laub_iterate_overlap_operator_squared(x):
            return self.gamma5_function(self.massive_Kenney_Laub_iterate_overlap_operator_linear_function(self.gamma5_function(self.massive_Kenney_Laub_iterate_overlap_operator_linear_function(x))))

        b_prime = self.gamma5_function(self.massive_Kenney_Laub_iterate_overlap_operator_linear_function(self.gamma5_function(b)))
        b_norm = np.linalg.norm(b_prime)

        r = b_prime
        # - massive_Kenney_Laub_iterate_overlap_operator_squared(x0)
        residue_norm = np.linalg.norm(r)

        p = r
        x = x0

        # counter = 0
        while True:

            residue_norm = np.linalg.norm(r)

            Adotp = massive_Kenney_Laub_iterate_overlap_operator_squared(p)

            denominator = np.absolute((np.conjugate(p)).dot(Adotp))

            alpha_factor = (residue_norm)**2/ denominator
            
            x = x + alpha_factor*p
            r = r - alpha_factor*Adotp

            if (np.linalg.norm(r)/b_norm < precision):
                break

            beta_factor = (np.linalg.norm(r) / residue_norm)**2
            p = r + beta_factor*p

            # counter += 1
        
        return x
    
    # 2ND ITERATION
    def inverse_Kenney_Laub_2nd_iterate_overlap_operator_function(self, precision, number_of_rows=4):

        x0 = np.zeros((self.fermion_dimensions*(self.lattice_size**self.lattice_dimensions),), dtype=np.complex_)
        b = np.zeros((self.fermion_dimensions*(self.lattice_size**self.lattice_dimensions),), dtype=np.complex_)
        b[0] = 1.0

        inverse_Kenney_Laub_2nd_iterate_overlap_operator = list()
        for next_permutation in range(number_of_rows):
            
            inverse_Kenney_Laub_2nd_iterate_overlap_operator_vector = self.Kenney_Laub_2nd_iterate_overlap_operator_functional_conjugate_gradient_solution_function(b, x0, precision)
            
            inverse_Kenney_Laub_2nd_iterate_overlap_operator.append(inverse_Kenney_Laub_2nd_iterate_overlap_operator_vector)

            b = np.roll(b, axis=0, shift=+1)

        inverse_Kenney_Laub_2nd_iterate_overlap_operator = (np.array(inverse_Kenney_Laub_2nd_iterate_overlap_operator))

        return inverse_Kenney_Laub_2nd_iterate_overlap_operator.T

    def massive_Kenney_Laub_2nd_iterate_overlap_operator_linear_function(self, x, precision=1E-4):

        return (1-0.5*self.bare_mass_value)*self.Kenney_Laub_2nd_iterate_overlap_operator_linear_function(x, precision) + self.bare_mass_value*x
    
    def Kenney_Laub_2nd_iterate_overlap_operator_linear_function(self, x, precision=1E-4):

        return x + self.gamma5_function(self.approximate_matrix_Kenney_Laub_2nd_iterate_sign_function(x, precision))

    def approximate_matrix_Kenney_Laub_2nd_iterate_sign_function(self, x, precision=1E-4):
        # Sign function

        def h_vector(x):
            return self.approximate_matrix_Kenney_Laub_iterate_sign_function(x, precision)
        
        x0 = np.zeros(np.shape(x), dtype=np.complex_)

        return (1./3.)*h_vector(x + (8./3.)*self.Kenney_Laub_2nd_iterate_denominator_inverse(x, x0, precision))

    def Kenney_Laub_2nd_iterate_denominator_inverse(self, b, x0, precision=1E-4):

        # helper functions
        def h_vector(x):
            return self.approximate_matrix_Kenney_Laub_iterate_sign_function(x, precision)
        
        def Kenney_Laub_2nd_iterate_denominator_operator(x):
            return h_vector(h_vector(x)) + (1./3.)*x

        b_norm = np.linalg.norm(b)

        r = b
        # - Kenney_Laub_2nd_iterate_denominator_operator(x0)
        residue_norm = np.linalg.norm(r)

        p = r
        x = x0

        # counter = 0
        while True:

            residue_norm = np.linalg.norm(r)

            Adotp = Kenney_Laub_2nd_iterate_denominator_operator(p)

            denominator = np.absolute((np.conjugate(p)).dot(Adotp))

            alpha_factor = (residue_norm)**2/ denominator
            
            x = x + alpha_factor*p
            r = r - alpha_factor*Adotp

            if (np.linalg.norm(r)/b_norm < precision):
                break

            beta_factor = (np.linalg.norm(r) / residue_norm)**2
            p = r + beta_factor*p

            # counter += 1
        
        return x

    def Kenney_Laub_2nd_iterate_overlap_operator_functional_conjugate_gradient_solution_function(self, b, x0, precision=1E-4):

        # helper function
        def massive_Kenney_Laub_2nd_iterate_overlap_operator_squared(x):
            return self.gamma5_function(self.massive_Kenney_Laub_2nd_iterate_overlap_operator_linear_function(self.gamma5_function(self.massive_Kenney_Laub_2nd_iterate_overlap_operator_linear_function(x))))

        b_prime = self.gamma5_function(self.massive_Kenney_Laub_2nd_iterate_overlap_operator_linear_function(self.gamma5_function(b)))
        b_norm = np.linalg.norm(b_prime)

        r = b_prime
        # - massive_Kenney_Laub_2nd_iterate_overlap_operator_squared(x0)
        residue_norm = np.linalg.norm(r)

        p = r
        x = x0

        # counter = 0
        while True:

            residue_norm = np.linalg.norm(r)

            Adotp = massive_Kenney_Laub_2nd_iterate_overlap_operator_squared(p)

            denominator = np.absolute((np.conjugate(p)).dot(Adotp))

            alpha_factor = (residue_norm)**2/ denominator
            
            x = x + alpha_factor*p
            r = r - alpha_factor*Adotp

            if (np.linalg.norm(r)/b_norm < precision):
                break

            beta_factor = (np.linalg.norm(r) / residue_norm)**2
            p = r + beta_factor*p

            # counter += 1
        
        return x

    # 3RD ITERATION
    def inverse_Kenney_Laub_3rd_iterate_overlap_operator_function(self, precision, number_of_rows=4):

        x0 = np.zeros((self.fermion_dimensions*(self.lattice_size**self.lattice_dimensions),), dtype=np.complex_)
        b = np.zeros((self.fermion_dimensions*(self.lattice_size**self.lattice_dimensions),), dtype=np.complex_)
        b[0] = 1.0

        inverse_Kenney_Laub_3rd_iterate_overlap_operator = list()
        for next_permutation in range(number_of_rows):
            
            inverse_Kenney_Laub_3rd_iterate_overlap_operator_vector = self.Kenney_Laub_3rd_iterate_overlap_operator_functional_conjugate_gradient_solution_function(b, x0, precision)
            
            inverse_Kenney_Laub_3rd_iterate_overlap_operator.append(inverse_Kenney_Laub_3rd_iterate_overlap_operator_vector)

            b = np.roll(b, axis=0, shift=+1)

        inverse_Kenney_Laub_3rd_iterate_overlap_operator = (np.array(inverse_Kenney_Laub_3rd_iterate_overlap_operator))

        return inverse_Kenney_Laub_3rd_iterate_overlap_operator.T

    def massive_Kenney_Laub_3rd_iterate_overlap_operator_linear_function(self, x, precision=1E-4):

        return (1-0.5*self.bare_mass_value)*self.Kenney_Laub_3rd_iterate_overlap_operator_linear_function(x, precision) + self.bare_mass_value*x
    
    def Kenney_Laub_3rd_iterate_overlap_operator_linear_function(self, x, precision=1E-4):

        return x + self.gamma5_function(self.approximate_matrix_Kenney_Laub_3rd_iterate_sign_function(x, precision))

    def approximate_matrix_Kenney_Laub_3rd_iterate_sign_function(self, x, precision=1E-4):
        # Sign function

        def h_vector(x):
            return self.approximate_matrix_Kenney_Laub_iterate_sign_function(x, precision)
        
        x0 = np.zeros(np.shape(x), dtype=np.complex_)

        return (1./3.)*h_vector(x + (8./3.)*self.Kenney_Laub_3rd_iterate_denominator_inverse(x, x0, precision))

    def Kenney_Laub_3rd_iterate_denominator_inverse(self, b, x0, precision=1E-4):

        # helper functions
        def h_vector(x):
            return self.approximate_matrix_Kenney_Laub_iterate_sign_function(x, precision)
        
        def Kenney_Laub_3rd_iterate_denominator_operator(x):
            return h_vector(h_vector(x)) + (1./3.)*x

        b_norm = np.linalg.norm(b)

        r = b
        # - Kenney_Laub_3rd_iterate_denominator_operator(x0)
        residue_norm = np.linalg.norm(r)

        p = r
        x = x0

        # counter = 0
        while True:

            residue_norm = np.linalg.norm(r)

            Adotp = Kenney_Laub_3rd_iterate_denominator_operator(p)

            denominator = np.absolute((np.conjugate(p)).dot(Adotp))

            alpha_factor = (residue_norm)**2/ denominator
            
            x = x + alpha_factor*p
            r = r - alpha_factor*Adotp

            if (np.linalg.norm(r)/b_norm < precision):
                break

            beta_factor = (np.linalg.norm(r) / residue_norm)**2
            p = r + beta_factor*p

            # counter += 1
        
        return x

    def Kenney_Laub_3rd_iterate_overlap_operator_functional_conjugate_gradient_solution_function(self, b, x0, precision=1E-4):

        # helper function
        def massive_Kenney_Laub_3rd_iterate_overlap_operator_squared(x):
            return self.gamma5_function(self.massive_Kenney_Laub_3rd_iterate_overlap_operator_linear_function(self.gamma5_function(self.massive_Kenney_Laub_3rd_iterate_overlap_operator_linear_function(x))))

        b_prime = self.gamma5_function(self.massive_Kenney_Laub_3rd_iterate_overlap_operator_linear_function(self.gamma5_function(b)))
        b_norm = np.linalg.norm(b_prime)

        r = b_prime
        # - massive_Kenney_Laub_3rd_iterate_overlap_operator_squared(x0)
        residue_norm = np.linalg.norm(r)

        p = r
        x = x0

        # counter = 0
        while True:

            residue_norm = np.linalg.norm(r)

            Adotp = massive_Kenney_Laub_3rd_iterate_overlap_operator_squared(p)

            denominator = np.absolute((np.conjugate(p)).dot(Adotp))

            alpha_factor = (residue_norm)**2/ denominator
            
            x = x + alpha_factor*p
            r = r - alpha_factor*Adotp

            if (np.linalg.norm(r)/b_norm < precision):
                break

            beta_factor = (np.linalg.norm(r) / residue_norm)**2
            p = r + beta_factor*p

            # counter += 1
        
        return x
    
    # GENERIC FUNCTIONS

    def generic_partial_operator_function(self, generic_operator_linear_function, parameters, number_of_rows=4):

        field_vector = np.zeros((self.fermion_dimensions*(self.lattice_size**self.lattice_dimensions),), dtype=np.complex_)
        field_vector[0] = 1.0

        partial_operator_array = list()
        for next_permutation in range(number_of_rows):

            partial_operator_vector = generic_operator_linear_function(field_vector, *parameters)
            
            partial_operator_array.append(partial_operator_vector)

            field_vector = np.roll(field_vector, axis=0, shift=+1)

        partial_operator_array = (np.array(partial_operator_array))

        return partial_operator_array.T

    def generic_inverse_of_operator_linear_function(self, operator_linear_function, x0, b, precision=1E-4):
        '''Application the of conjugate gradient algorithm'''

        # helper function
        def operator_squared_function(x):
            return self.gamma5_function(operator_linear_function(self.gamma5_function(operator_linear_function(x))))
        

        b_prime = self.gamma5_function(operator_linear_function(self.gamma5_function(b)))
        b_norm = np.linalg.norm(b_prime)

        r = b_prime
        # - operator_squared_function(x0)
        residue_norm = np.linalg.norm(r)

        p = r
        x = x0

        counter = 0
        while True:

            residue_norm = np.linalg.norm(r)

            Adotp = operator_squared_function(p)

            denominator = np.absolute((np.conjugate(p)).dot(Adotp))

            print(denominator)

            if (denominator == 0.):
                break
            
            alpha_factor = (residue_norm)**2/ denominator

            x = x + alpha_factor*p
            r = r - alpha_factor*Adotp

            if (np.linalg.norm(r)/b_norm < precision):
                break

            beta_factor = (np.linalg.norm(r) / residue_norm)**2
            p = r + beta_factor*p

            counter += 1
        
        return x
    
    def generic_massive_overlap_operator_linear_function(self, generic_sign_function_linear_function, parameters, x):

        return (1-0.5*self.bare_mass_value)*self.generic_overlap_operator_linear_function(generic_sign_function_linear_function, parameters, x) + self.bare_mass_value*x
    
    def generic_overlap_operator_linear_function(self, generic_sign_function_linear_function, parameters, x):
        '''Massless overlap operator function'''

        return x + self.gamma5_function(generic_sign_function_linear_function(*parameters, x))
    
    # def extremum_eigenvalues(self):
    #     # Calculate the eigenvalues of the square of the kernel
    #         class ExplicitLinearOperatorClass(LinearOperator):
    #             def __init__(self, N, dtype='float32'):
    #                 self.N = N
    #                 self.shape = (self.N, self.N)
    #                 self.dtype = np.dtype(dtype)
    #             def kernel_function(self, x):
    #                 return self.improved_Wilson_operator_linear_function(x) - x
    #             def _matvec(self, x):
    #                 return self.gamma5_function(kernel_function(self.gamma5_function(kernel_function(x))))

    #         N = self.fermion_dimensions*(self.lattice_size**self.lattice_dimensions)

    #         linear_operator_object = ExplicitLinearOperatorClass(N, dtype=np.complex_)

    #         largest_eigenvalue = eigsh(linear_operator_object, k=1, which = 'LM', tol=1E-5)[0]
    #         smallest_eigenvalue = eigsh(linear_operator_object, k=1, which = 'SM', tol=1E-5)[0]

    #         alpha = np.sqrt(smallest_eigenvalue[0])
    #         beta = np.sqrt(largest_eigenvalue[0])

    #     return alpha, beta

    