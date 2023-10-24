import numpy as np
import itertools
from mpi4py import MPI
import itertools as it


class WriteCoordinateError(Exception):
    pass


class LatticeStructure:
    '''This class has two constructors for setting the immutable fundamental parameters of the lattice.'''

    def __init__(self, lattice_size=9, lattice_dimensions=2, fermion_dimensions=4, *, temporal_axis_size=None):
        '''The assumption is that the lattice has the same size in all direction. Passing optionally a "temporal_axis_size" value indicates that the temporal direction only has a different size.'''

        assert isinstance(lattice_size, int) and (lattice_size >= 9), 'Lattice size must be a positive integer greater or equal to 9.'
        self._lattice_size = lattice_size

        assert isinstance(lattice_dimensions, int) and lattice_dimensions in [1, 2, 3, 4], 'Lattice dimensions must be a positive integer greater or equal to 1 and less or equal to 4.'
        self._lattice_dimensions = lattice_dimensions

        assert isinstance(fermion_dimensions, int) and fermion_dimensions in [2, 4], 'Fermion dimensions must be a positive integer equal either to 2 or 4.'
        self._fermion_dimensions = fermion_dimensions

        if temporal_axis_size:
            assert isinstance(temporal_axis_size, int) and (temporal_axis_size >= 9), 'Lattice size must be a positive integer greater or equal to 9.'
            self._lattice_shape = tuple(temporal_axis_size, *([lattice_size]*(lattice_dimensions-1)))
        else:
            self._lattice_shape = tuple([lattice_size]*lattice_dimensions)

        '''
        TODO: Rethink whether to keep the communicator rank information in the lattice class.
        '''
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

    '''
    TODO: Should I introduce a "fundamental_variables" dictionary
    
    '''
    
    @classmethod
    def from_gauge_links_field_array(cls, lattice_shape=(9,9), fermion_dimensions=4):
        '''Alternatively, the shape of the lattice can be passed explicitly. By convention axis 0 corresponds to the temporal direction and the rest of the axes to the spatial directions.'''

        temporal_axis_size = lattice_shape[0]
        lattice_size = lattice_shape[1]
        lattice_dimensions = len(lattice_shape)

        return cls(lattice_size, lattice_dimensions, fermion_dimensions, temporal_axis_size)

    @property
    def lattice_size(self):
        return self._lattice_size
    
    @lattice_size.setter
    def lattice_size(self, lattice_size):
        raise WriteCoordinateError("The lattice size is a fundamental parameter of the lattice structure and cannot be modified.")

    @property
    def lattice_dimensions(self):
        return self._lattice_dimensions
    
    @lattice_dimensions.setter
    def lattice_dimensions(self, lattice_dimensions):
        raise WriteCoordinateError("The lattice dimension is a fundamental parameter of the lattice structure and cannot be modified.")

    @property
    def fermion_dimensions(self):
        return self._fermion_dimensions
    
    @fermion_dimensions.setter
    def fermion_dimensions(self, fermion_dimensions):
        raise WriteCoordinateError("The fermion dimension is a fundamental parameter of the lattice structure and cannot be modified.")
    
    @property
    def lattice_shape(self):
        return self._lattice_shape
    
    @lattice_shape.setter
    def lattice_shape(self, lattice_shape):
        raise WriteCoordinateError("The lattice shape is a fundamental parameter of the lattice structure and cannot be modified.")

    def __repr__(self) -> str:
        return f"\n{type(self).__name__}(lattice_size={self._lattice_size!r}, lattice_dimensions={self._lattice_dimensions!r}, fermion_dimensions={self._fermion_dimensions!r}, temporal_axis_size={self._lattice_shape[0]!r})\n"

    def __str__(self) -> str:
        return f'\nA {self._lattice_dimensions}D lattice structure of shape ({self._lattice_shape}) has been constructed accommodating {self._fermion_dimensions}-component fermions.\n'
    
    def lattice_sites_coordinates_addition(self, a, b):
        '''-INPUT: two tuples of size self.lattice_dimensions.
        -USAGE: a + b plus periodic boundary condition for any lattice size and dimensions.
        -OUTPUT: tuple of size self.lattice_dimensions.
        -NOTE: performance tested against iterator tools and still faired time-wise better.'''

        lattice_sites_coordinates_sum = tuple()
        for index in range(self.lattice_dimensions):
            lattice_sites_coordinates_sum += ( ((a[index]+b[index]+self.lattice_size)%self.lattice_size), )
        
        return lattice_sites_coordinates_sum


##########################################
##########################################
##########################################
##########################################
##########################################


    '''
    TODO: Remove the following methods
    '''

    def coordinate_index(self, coordinate_tuple):

        coordinate_index = 0
        for i in range(self.lattice_dimensions):
            coordinate_tuple[i]*self.lattice_size**(self.lattice_dimensions-1-i)

        # return coordinate_tuple[0]*self.lattice_size + coordinate_tuple[1]

        return coordinate_index

    def lattice_sites_coordinates(self):
        ''' OUTPUT: an array of size (lattice_size^2, lattice_dimensions)
        TODO:
        * document usage
        * parallel case
        '''
        
        list_of_axes = [[flat_index for flat_index in range(self.lattice_size)] ]*self.lattice_dimensions
        
        return list(itertools.product(*list_of_axes))
    