'''
TODO: denote properly public and private members of classes
'''

import numpy as np
import itertools
from mpi4py import MPI


class LatticeStructure:

    def __init__(self, lattice_size=9, lattice_dimensions=2, fermion_dimensions=4):

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        assert isinstance(lattice_size, int) and (lattice_size >= 9), 'Lattice size must be a positive integer greater or equal to 9.'
        self.lattice_size = lattice_size

        # assert lattice_size%size == 0, 'The number of processes must be a divisor of the lattice size.'
        self.comm = comm
        self.rank = rank
        self.size = size

        assert isinstance(lattice_dimensions, int) and fermion_dimensions in [1, 2, 3, 4], 'Lattice dimensions must be a positive integer greater or equal to 1 and less or equal to 4.'
        self.lattice_dimensions = lattice_dimensions

        assert isinstance(fermion_dimensions, int) and fermion_dimensions in [2, 4], 'Fermion dimensions must be a positive integer equal either to 2 or 4.'
        self.fermion_dimensions = fermion_dimensions

    def __repr__(self) -> str:

        # TODO Anticipate the case the size of the time direction differs from the spatial ones.
        
        lattice_shape = tuple([self.lattice_size]*self.lattice_dimensions)

        return f"\nA {self.lattice_dimensions}D lattice structure of shape ({lattice_shape}) has been initialized accommodating {self.fermion_dimensions}-component fermions.\n"
        # {type(self).__name__}

    def lattice_sites_coordinates(self):
        ''' OUTPUT: an array of size (lattice_size^2, lattice_dimensions)
        TODO:
        * document usage
        * parallel case
        '''
        
        list_of_axes = [[flat_index for flat_index in range(self.lattice_size)] ]*self.lattice_dimensions
        
        return list(itertools.product(*list_of_axes))
    
    def tuple_addition(self, a, b):

        lattice_size = self.lattice_size

        result = ((a[0]+b[0]+lattice_size)%lattice_size, (a[1]+b[1]+lattice_size)%lattice_size)
        
        return result
    
    def coordinate_index(self, coordinate_tuple):
        
        return coordinate_tuple[0]*self.lattice_size + coordinate_tuple[1]

###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################

class GaugeLinksField(LatticeStructure):

    def __init__(self, lattice_size, lattice_dimensions, fermion_dimensions, theory_label=1, random_gauge_links_field_boolean=False):

        super().__init__(lattice_size, lattice_dimensions, fermion_dimensions)

        assert isinstance(theory_label, int) and theory_label in (1, 2, 3), 'Theory label must be a positive integer equal either to 1, 2 or 3 indicating U(1), SU(2), SU(3) gauge links fields correspondingly.'
        self.theory_label = theory_label

        # Create a random phase values array if requested
        if random_gauge_links_field_boolean:
            self._gauge_links_phase_values_field_array = self.phase_values_array_random_generator()
    
    @classmethod
    def from_LatticeStructure_object(cls, lattice_structure, theory_label):

        assert isinstance(lattice_structure, LatticeStructure), 'The argument must be an instance of the "LatticeStructure" class.'
        # cls.lattice_structure = lattice_structure

        return cls(lattice_structure.lattice_size, lattice_structure.lattice_dimensions, lattice_structure.fermion_dimensions, theory_label)
    
    @classmethod
    def from_gauge_links_field_array(cls, gauge_links_field_array, fermion_dimensions=4):

        '''
        TODO:
        * Check if gauge_links_field_array is not empty
        * Improve method for extracting lattice_size, lattice_dimensions, and theory_label.
        '''

        gauge_links_field_array = np.array(gauge_links_field_array)

        gauge_links_field_array_shape = np.shape(gauge_links_field_array)

        lattice_size = gauge_links_field_array_shape[0]
        lattice_dimensions = gauge_links_field_array_shape[-1]
        theory_label = 1

        return cls(lattice_size, lattice_dimensions, fermion_dimensions, theory_label)
            
    def __repr__(self) -> str:
        '''
        TODO:
        * use the LatticeStructure __repr__
        * check existence of gauge links field attribute
        '''
        # print(self.__repr__)
        
        lattice_shape = tuple([self.lattice_size]*self.lattice_dimensions)
        if (self.theory_label == 1):
            gauge_theory_label = 'U(1)'
        else:
            gauge_theory_label = 'SU('+str(self.theory_label)+')'

        return  f"\nA {self.lattice_dimensions}D lattice structure of shape {lattice_shape} has been initialized accommodating {self.fermion_dimensions}-component fermions.\nIn addition, embedded in this lattice structure is a {gauge_theory_label} gauge links field of shape {np.shape(self.gauge_links_phase_values_field_array)}.\n"

        # return f'\nAn instance of the {type(self).__name__} class has been created with arguments: lattice_size={self.lattice_size}, lattice_dimensions={self.lattice_dimensions}, fermion_dimensions={self.fermion_dimensions}, theory_label={self.theory_label}.\n'

    '''
    TODO: Validate input with assert statements
    '''
    @property
    def gauge_links_phase_values_field_array(self):
        return self._gauge_links_phase_values_field_array
    
    @gauge_links_phase_values_field_array.setter
    def gauge_links_phase_values_field_array(self, gauge_links_phase_values_field_array):
        '''
        TODO: provide a rudimentary check for gauge_links_phase_values_field_array
        '''
        self._gauge_links_phase_values_field_array = gauge_links_phase_values_field_array

    '''
    TODO: gauge_links_field_array is NOT private. It's a function of gauge_links_phase_values_field_array. 
    '''
    @property #DO NOT USE IT!
    def gauge_links_field_array(self):
        return self.gauge_links_field_array
    
    @gauge_links_field_array.setter
    def gauge_links_field_array(self, gauge_links_field_array):
        # The gauge links field values are used to calculate thw corresponding phase values field
        '''
        TODO: Check input: shape and type.
        '''
        self._gauge_links_phase_values_field_array = np.real(1.j*np.log(gauge_links_field_array))

    def phase_values_array_random_generator(self, seed=None, phase_values_range=np.pi/2.0):
        '''
        * INPUT: phase values range variable corresponds to the difference between the maximum minus the minimum phase value of φ in the exp(iφ) expression
        * OUTPUT: array of shape (lattice_size, lattice_size, lattice_dimensions)
        '''
        
        '''
        TODO: 
        * configure random generator
        * uniform or gaussian distribution
        '''
        
        # Use of the default constructor for the Generator class, equivalent to: np.random.default_rng(seed) 
        # random_generator = np.random.default_rng(seed) 
        # np.random.Generator(np.random.PCG64(seed))
        
        phase_values_array_dimensions = [*[self.lattice_size]*self.lattice_dimensions, self.lattice_dimensions]        
        if (self.theory_label != 1):
            phase_values_array_dimensions += [self.theory_label]*2

        random_phase_values_array = np.random.rand(*phase_values_array_dimensions)
        random_phase_values_array = 2*phase_values_range*random_phase_values_array - phase_values_range

        return random_phase_values_array
        
    def links_values_array_random_generator(self, seed=None, phase_values_range=np.pi/2.0):
        '''
        TODO: adjust for parallel usage
        '''

        random_phase_values_array = self.phase_values_array_random_generator(seed, phase_values_range)
        
        links_values_array_random_array = np.exp(random_phase_values_array*1.j)

        return links_values_array_random_array
    
    def gauge_links_phase_values_field_array_function(self, gauge_links_phase_values_field_array):
        '''
        TODO: adjust for parallel usage
        '''

        gauge_links_phase_values_field_array = np.array(gauge_links_phase_values_field_array)

        assert all(isinstance(element, float) for element in gauge_links_phase_values_field_array.reshape(-1)), 'The elements of the gauge links field array must be real numbers.'

        expected_array_dimensions = [*[self.lattice_size]*self.lattice_dimensions, self.lattice_dimensions]
        if (self.theory_label != 1):
            expected_array_dimensions += [self.theory_label, self.theory_label]
        assert np.shape(gauge_links_phase_values_field_array) == tuple(expected_array_dimensions), 'A multidimensional array is expected of shape (L, ..., L, d, n, n).'

        self.gauge_links_phase_values_field_array = gauge_links_phase_values_field_array

        return print(f'\nAn array of shape {np.shape(self.gauge_links_phase_values_field_array)} has been passed for the phase values of the gauge links field.\n')
        
    def gauge_links_field_array_function(self, gauge_links_field_array):
        '''
        TODO: adjust for parallel usage
        '''

        gauge_links_field_array = np.array(gauge_links_field_array)

        assert all(isinstance(element, complex) for element in gauge_links_field_array.reshape(-1)), 'The elements of the gauge links field array must be complex numbers.'

        expected_array_dimensions = [*[self.lattice_size]*self.lattice_dimensions, self.lattice_dimensions]
        if (self.theory_label != 1):
            expected_array_dimensions += [self.theory_label, self.theory_label]
        assert np.shape(gauge_links_field_array) == tuple(expected_array_dimensions), 'A multidimensional array is expected of shape (L, ..., L, d, n, n).'

        self.gauge_links_field_array = gauge_links_field_array
        
        return print(f'\nAn array of shape {np.shape(self.gauge_links_field_array)} has been passed as the gauge links field.\n')

    def gauge_links_field_function(self, gauge_links_phase_values_field, direction):
        '''
        TODO:
        * Write description
        * Adjust to parallel case
        '''

        # directions: (0, +1) & (+1, 0)
        if (np.sum(direction) == +1):
            gauge_links_field_array = np.exp( 1.j*((gauge_links_phase_values_field.T)[abs(direction[0])]).T)

        # directions: (0, -1) & (-1, 0)
        elif (np.sum(direction) == -1):
            shifted_U1_gauge_links_phase_values_field_array = (-1.0)*np.roll(gauge_links_phase_values_field, axis=abs(direction[1]), shift=+1)
            gauge_links_field_array = np.exp( 1.j*((shifted_U1_gauge_links_phase_values_field_array.T)[abs(direction[0])]).T)

        # central terms
        elif (direction == (0,0)):
            # matrix_dimension = np.shape(gauge_links_phase_values_field)[0]
            # gauge_links_field_array = np.ones((matrix_dimension, matrix_dimension), dtype=np.complex_)
            gauge_links_field_array = np.ones(np.shape(gauge_links_phase_values_field)[:-1], dtype=np.complex_)

        # diagonal terms
        else:
            matrix_dimension = np.shape(gauge_links_phase_values_field)[0]
            # gauge_links_field_array = np.zeros((matrix_dimension, matrix_dimension), dtype=np.complex_)
            gauge_links_field_array = np.zeros(np.shape(gauge_links_phase_values_field)[:-1], dtype=np.complex_)
            
            for path_index in [0, 1]:
                # 1st piece
                shifted_U1_gauge_links_phase_values_field_array = np.roll(gauge_links_phase_values_field, axis=(path_index+1)%2, shift=(-direction[(path_index+1)%2]+1)//2)
                path_specific_gauge_links_phase_values_field = direction[(path_index+1)%2]*((shifted_U1_gauge_links_phase_values_field_array.T)[path_index]).T
                # 2nd piece
                shifted_U1_gauge_links_phase_values_field_array = np.roll(
                    np.roll(gauge_links_phase_values_field, axis=(path_index+1)%2, shift= -direction[(path_index+1)%2]),
                    axis=path_index, shift=(-direction[path_index]+1)//2)
                path_specific_gauge_links_phase_values_field += direction[path_index]*((shifted_U1_gauge_links_phase_values_field_array.T)[(path_index+1)%2]).T
                
                gauge_links_field_array += np.exp( 1.j*path_specific_gauge_links_phase_values_field)

            gauge_links_field_array = gauge_links_field_array/np.absolute(gauge_links_field_array)

        return gauge_links_field_array
