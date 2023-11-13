'''This module contains "GaugeLinksField" a subclass of "LatticeStructure" that contains the actual gauge links field values and serves also as a base class for all the operator-related classes.'''
import numpy as np

import custom_library.auxiliary as auxiliary
import custom_library.lattice as lattice


class GaugeLinksField(lattice.LatticeStructure):

    def __init__(self, gauge_links_field_array, gauge_links_phase_value_field_boolean=False, fermion_dimensions=4, *, random_gauge_links_field_boolean=False):
        '''Expected shapes for the gauge_links_field_array:
        - 1D: (L,) or (L,u,u)
        - 2D: (L,L,2) or (L,L,2,u,u)
        - 3D: (L,L,L,3) or (L,L,L,3,u,u)
        - 4D: (L,L,L,L,4) or (L,L,L,L,4,u,u)
        '''
        
        assert isinstance(random_gauge_links_field_boolean, bool), 'Parameter "random_gauge_links_field_boolean" takes only boolean values, True or False.'
        self._random_gauge_links_field_boolean = random_gauge_links_field_boolean # Stored only for debugging purposes.

        gauge_links_field_array = np.array(gauge_links_field_array)
        assert (gauge_links_field_array.size != 0) and (gauge_links_field_array.ndim >= 1), 'Passed gauge links field array must be a non-empty multidimensional array.'

        # Extracting the "gauge_theory_label" fundamental immutable parameter, along with the lattice shape for the initialization of the base class.
        gauge_links_field_array_shape = np.shape(gauge_links_field_array)
        # SU(2) and SU(3) cases 
        if (gauge_links_field_array_shape[-1] == gauge_links_field_array_shape[-2]):
            assert self._gauge_theory_label in [2, 3], 'Gauge group theory groups used are not among the expected U(1), SU(2), or SU(3) ones.'
            self._gauge_theory_label = gauge_links_field_array_shape[-1]
             # This covers immediately the 1D case
            lattice_shape = gauge_links_field_array_shape[:-2]

        else: # U(1) case
            self._gauge_theory_label = 1
            lattice_shape = gauge_links_field_array_shape # 1D case
        
        if (len(lattice_shape) >= 3): # Rest 2D, 3D, and 4D cases
            # Additional check
            assert lattice_shape[-1] == len(lattice_shape) - 1, 'Lattice dimensions do not correspond to the number of lattice axes.'
            lattice_shape = lattice_shape[:-1]

        lattice_size, lattice_dimensions, temporal_axis_size = lattice.LatticeStructure.turning_lattice_shape_to_fundamental_parameters(lattice_shape=lattice_shape)

        super().__init__(lattice_size=lattice_size, lattice_dimensions=lattice_dimensions, fermion_dimensions=fermion_dimensions, temporal_axis_size=temporal_axis_size)
        
        # Differentiating between two cases: whether the "gauge_links_field_array" contains actual gauge link values at lattice sites or if it instead contains gauge link phase values.
        assert isinstance(gauge_links_phase_value_field_boolean, bool), 'Parameter "gauge_links_phase_value_field_boolean" takes only boolean values, True or False.'
        self._gauge_links_phase_value_field_boolean = gauge_links_phase_value_field_boolean

        if gauge_links_phase_value_field_boolean: # Phase values
            assert np.all(np.isreal(gauge_links_field_array)), 'The gauge links phase values array passed must be real-valued.'
            self._gauge_links_phase_values_field_array = gauge_links_field_array
            # Constructing the corresponding gauge links field array
            self._gauge_links_field_array = self._converting_to_gauge_links_field_function()
        
        else: # Actual gauge links values at lattice sites
            assert np.any(np.iscomplex(gauge_links_field_array)), 'The gauge links array passed must be complex-valued.'
            self._gauge_links_field_array = gauge_links_field_array
            # Constructing the corresponding gauge links phase values field array
            self._gauge_links_phase_values_field_array = self._converting_to_gauge_links_phase_values_field_function()
    
    @classmethod
    def from_lattice_shape_with_random_gauge_links_field(cls, lattice_shape=(9,9), fermion_dimensions=4, gauge_theory_label=1, random_generator_seed=None, random_phase_values_range=np.pi/2):
        '''NOTE: Alternative constructor ONLY for testing purposes.'''
        
        # A random array is constructed
        gauge_links_phase_value_field_array = cls.random_gauge_links_field_phase_values_array_function(lattice_shape=lattice_shape, gauge_theory_label=gauge_theory_label, random_generator_seed=random_generator_seed, random_phase_values_range=random_phase_values_range)

        return cls(gauge_links_field_array=gauge_links_phase_value_field_array, gauge_links_phase_value_field_boolean=True, fermion_dimensions=fermion_dimensions, random_gauge_links_field_boolean=True)

    @property
    def gauge_theory_label(self):
        return self._gauge_theory_label
    
    @gauge_theory_label.setter
    def gauge_theory_label(self, gauge_theory_label):
        raise auxiliary.ReadOnlyAttributeError(gauge_theory_label=gauge_theory_label)

    @property
    def gauge_links_field_array(self):
        return self._gauge_links_field_array
    
    @gauge_links_field_array.setter
    def gauge_links_field_array(self, gauge_links_field_array):
        raise auxiliary.ReadOnlyAttributeError(gauge_links_field_array=gauge_links_field_array)
    
    @property
    def gauge_links_phase_values_field_array(self):
        return self._gauge_links_phase_values_field_array
    
    @gauge_links_phase_values_field_array.setter
    def gauge_links_phase_values_field_array(self, gauge_links_phase_values_field_array):
        raise auxiliary.ReadOnlyAttributeError(gauge_links_phase_values_field_array=gauge_links_phase_values_field_array)

    @property
    def gauge_links_phase_value_field_boolean(self):
        return self._gauge_links_phase_value_field_boolean
    
    @gauge_links_phase_value_field_boolean.setter
    def gauge_links_phase_value_field_boolean(self, gauge_links_phase_value_field_boolean):
        raise auxiliary.ReadOnlyAttributeError(gauge_links_phase_value_field_boolean=gauge_links_phase_value_field_boolean)
    
    @property
    def random_gauge_links_field_boolean(self):
        return self._random_gauge_links_field_boolean
    
    @random_gauge_links_field_boolean.setter
    def random_gauge_links_field_boolean(self, random_gauge_links_field_boolean):
        raise auxiliary.ReadOnlyAttributeError(random_gauge_links_field_boolean=random_gauge_links_field_boolean)
    
    def __repr__(self) -> str:
        '''The output depends on whether *random* gauge links field was requested and if not, whether a gauge links field or a gauge links phase values field array was passed.'''

        if (not self._random_gauge_links_field_boolean):
            if not self._gauge_links_phase_value_field_boolean:
                return f'\n{type(self).__name__}(gauge_links_field_array={id(self._gauge_links_field_array)}, gauge_links_phase_value_field_boolean={self._gauge_links_phase_value_field_boolean!r}, fermion_dimensions={self._fermion_dimensions!r})\n'
            
            else:
                return f'\n{type(self).__name__}(gauge_links_field_array={id(self._gauge_links_phase_values_field_array)}, gauge_links_phase_value_field_boolean={self._gauge_links_phase_value_field_boolean!r}, fermion_dimensions={self._fermion_dimensions!r})\n'
        
        else:
            return f'\n{type(self).__name__}(gauge_links_field_array={id(self._gauge_links_phase_values_field_array)}, gauge_links_phase_value_field_boolean={self._gauge_links_phase_value_field_boolean!r}, fermion_dimensions={self._fermion_dimensions!r}, random_gauge_links_field_boolean={self._random_gauge_links_field_boolean!r})\n'
        
    def __str__(self) -> str:
        '''The str output of the base class is expanded with additional details about the embedded gauge links field.'''

        str_output = super().__str__()

        gauge_theory_label_string = f'U({self._gauge_theory_label})'
        if (self.gauge_theory_label != 1):
            gauge_theory_label_string = 'S'+gauge_theory_label_string

        str_output += f'Embedded in this lattice structure is a {gauge_theory_label_string} gauge links field.\n'

        return str_output

    @classmethod
    def random_gauge_links_field_phase_values_array_function(cls, lattice_shape=(9,9), gauge_theory_label=1, random_generator_seed=None, random_phase_values_range=np.pi/2):
        '''This is a special public method intended to be used only for testing purposes.'''

        _, lattice_dimensions, _ = lattice.LatticeStructure.turning_lattice_shape_to_fundamental_parameters(lattice_shape=lattice_shape)

        assert isinstance(gauge_theory_label, int) and (not isinstance(gauge_theory_label, bool)) and (gauge_theory_label in [1, 2, 3]), 'Input parameter "gauge_theory_label" must be a positive integer of value either 1, 2, or 3, corresponding to the U(1), SU(2), SU(3) gauge theory groups.'
        
        # Calculating the shape of the output array.
        links_field_phase_values_array_shape = lattice_shape
        # Appending lattice dimensions, apart from the 1D case
        if (lattice_dimensions != 1):
            links_field_phase_values_array_shape += (lattice_dimensions,)
        # Appending gauge group matrix shape, apart from the U(1) case
        if (gauge_theory_label != 1):
            links_field_phase_values_array_shape += (gauge_theory_label, gauge_theory_label)

        # Generating a random matrix from a uniform [0,1] distribution
        random_gauge_links_field_phase_values_array = np.random.rand(*links_field_phase_values_array_shape)
        
        # Shifting its values to adjust to the desired range of values
        ''''
        TODO:
        1. Make use of the PRG seed
        2. Check what's the recommended way of generating uniform elements of array in a specified range.
        '''
        # Anticipating the case for which an integer number is passed to the "random_phase_values_range" parameter
        if isinstance(random_phase_values_range, int) and (not isinstance(random_phase_values_range, bool)):
            random_phase_values_range = float(random_phase_values_range)
        assert isinstance(random_phase_values_range, float) and (random_phase_values_range>= 0.) and (random_phase_values_range<= np.pi/2), 'Input "random_phase_values_range" parameter must be a non-negative real-valued number smaller than or equal to π/2.'
        random_gauge_links_field_phase_values_array = 2*random_phase_values_range*random_gauge_links_field_phase_values_array - random_phase_values_range

        return random_gauge_links_field_phase_values_array

    def _converting_to_gauge_links_phase_values_field_function(self):
        '''φ = i*log(u)'''

        if (self._gauge_theory_label == 1):
            return np.real(-1.j*np.log(self._gauge_links_field_array))
        else:
            '''
            TODO: Construct the SU(2) and SU(3) case
            '''
            pass
        
    def _converting_to_gauge_links_field_function(self):
        '''u = exp(iφ)'''

        if (self._gauge_theory_label == 1):
            return np.exp(1.j*self._gauge_links_phase_values_field_array)
        else:
            '''
            TODO: Construct the SU(2) and SU(3) case
            '''
            pass
        