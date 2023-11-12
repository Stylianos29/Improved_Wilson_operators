'''This module contains the "LatticeStructure" class that serves as the base class for all the operator-related classes.'''
from mpi4py import MPI

import custom_library.auxiliary as auxiliary


class LatticeStructure:
    '''
    Represents lattice structures for Gauge and Fermion fields in Lattice QCD analysis.

    The term "lattice structure" refers to the shape of the lattice, without storing an array at this stage.
    It is assumed to be a tuple of components, all of the same size, or with the temporal direction being different.

    This class introduces the concept of "fundamental parameters" for the problem. Subclasses expand this list
    with specific immutable parameters. The fundamental parameters stored by objects of this class include:

    - `lattice_size`: Corresponds to the spatial directions of the lattice (unless it's the same for the temporal direction).
    - `lattice_dimensions`: Dimensions of the lattice.
    - `fermion_dimensions`: Dimensions related to fermion fields.
    - `lattice_shape`: The overall shape of the lattice.
    '''

    def __init__(self, lattice_size=9, lattice_dimensions=2, fermion_dimensions=4, *, temporal_axis_size=None):
        '''-NOTE: This constructor assumes a lattice with equal size in all directions. For cases with a different size in the temporal direction, use the alternative constructor.'''

        assert isinstance(lattice_size, int) and (lattice_size >= 9), 'Lattice size must be a positive integer greater or equal to 9.'
        self._lattice_size = lattice_size

        assert isinstance(lattice_dimensions, int) and (not isinstance(lattice_dimensions, bool)) and (lattice_dimensions in [1, 2, 3, 4]), 'Lattice dimensions must be a positive integer greater or equal to 1 and less or equal to 4.'
        self._lattice_dimensions = lattice_dimensions

        assert isinstance(fermion_dimensions, int) and (fermion_dimensions in [2, 4]), 'Fermion dimensions must be a positive integer equal either to 2 or 4.'
        self._fermion_dimensions = fermion_dimensions

        if temporal_axis_size is not None:
            assert isinstance(temporal_axis_size, int) and (temporal_axis_size >=9), 'Temporal axis size must be a positive integer greater or equal to 9.'
            self._lattice_shape = (temporal_axis_size,) + tuple([self._lattice_size]*(self._lattice_dimensions-1))
        
        else:
            self._lattice_shape = tuple([self._lattice_size]*self._lattice_dimensions)

        '''
        TODO: Rethink whether to keep the communicator rank information in the lattice class.
        '''
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

    @classmethod
    def from_lattice_shape(cls, lattice_shape=(9,9), fermion_dimensions=4):
        '''-USAGE: Alternative constructor; explicitly pass the lattice shape. Convention: Axis 0 is temporal; the rest are spatial. Assumes only the temporal axis differs in size. Example: an input lattice shape: (20, 2, 3, 10) will be taken to be as (20, 10, 10, 10)'''

        assert isinstance(lattice_shape, tuple) and (len(lattice_shape) > 0) and (len(lattice_shape) <= 4), 'Lattice shape must be tuple of at most 4 integer number all of them greater than or equal to 9.'
        lattice_size = lattice_shape[-1]
        lattice_dimensions = len(lattice_shape)
        temporal_axis_size = lattice_shape[0]

        return cls(lattice_size, lattice_dimensions, fermion_dimensions, temporal_axis_size=temporal_axis_size)

    @property
    def lattice_size(self):
        return self._lattice_size
    
    @lattice_size.setter
    def lattice_size(self, lattice_size):
        raise auxiliary.ReadOnlyAttributeError(lattice_size=lattice_size)

    @property
    def lattice_dimensions(self):
        return self._lattice_dimensions
    
    @lattice_dimensions.setter
    def lattice_dimensions(self, lattice_dimensions):
        raise auxiliary.ReadOnlyAttributeError(lattice_dimensions=lattice_dimensions)
    
    @property
    def fermion_dimensions(self):
        return self._fermion_dimensions
    
    @fermion_dimensions.setter
    def fermion_dimensions(self, fermion_dimensions):
        raise auxiliary.ReadOnlyAttributeError(fermion_dimensions=fermion_dimensions)
    
    @property
    def lattice_shape(self):
        return self._lattice_shape
    
    @lattice_shape.setter
    def lattice_shape(self, lattice_shape):
        raise auxiliary.ReadOnlyAttributeError(lattice_shape=lattice_shape)

    def __repr__(self) -> str:
        return f'\n{type(self).__name__}(lattice_size={self._lattice_size!r}, lattice_dimensions={self._lattice_dimensions!r}, fermion_dimensions={self._fermion_dimensions!r}, temporal_axis_size={self._lattice_shape[0]!r})'

    def __str__(self) -> str:
        return f'\nA {self._lattice_dimensions}D lattice structure of shape ({self._lattice_shape}) has been constructed accommodating {self._fermion_dimensions}-component fermions.\n'
    
    def lattice_coordinates_vectors_addition(self, tuple_a, tuple_b):
        '''-USAGE: Addition of a and b lattice coordinates vectors such that the periodic boundary condition is satisfied.
        -OUTPUT: tuple of size self_.lattice_dimensions.
        -NOTE: performance tested against iterator tools and still faired better time-wise.'''

        assert isinstance(tuple_a, tuple) and isinstance(tuple_b, tuple) and len(tuple_a) == len(tuple_b) and len(tuple_a) == self._lattice_dimensions, 'Input vectors for addition must be tuples of the same size equal to the lattice dimensions.'

        assert all(component_of_tuple_a < component_of_lattice_shape for component_of_tuple_a, component_of_lattice_shape in zip(tuple_a, self._lattice_shape)) and all(component_of_tuple_b < component_of_lattice_shape for component_of_tuple_b, component_of_lattice_shape in zip(tuple_a, self._lattice_shape)), 'Input vectors for addition must be tuples of integers, each component being smaller than the corresponding lattice shape component.'

        lattice_shape = self._lattice_shape

        lattice_coordinates_vectors_addition = tuple()
        for index in range(self.lattice_dimensions):
            lattice_coordinates_vectors_addition += ( ((tuple_a[index]+tuple_b[index]+lattice_shape[index])%lattice_shape[index]), )
                        
        return lattice_coordinates_vectors_addition
