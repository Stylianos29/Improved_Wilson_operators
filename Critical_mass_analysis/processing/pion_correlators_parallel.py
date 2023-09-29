'''
TODO: Write a more detailed description.
...
Parallel calculation of a number of time-dependent pion correlator arrays. The number is constraint by the number of processes requested.'''
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import LinearOperator
import h5py
import click
import sys
import os.path
from mpi4py import MPI
import time
from datetime import datetime

# custom modules
sys.path.append('..')
import custom_library.auxiliary as auxiliary
import custom_library.gauge as gauge
import custom_library.operators as operators
import custom_library.constants as constants


@click.command()
@click.option("--input_file", "input_file", "-in_file", default='/nvme/h/cy22sg1/Improved_Wilson_operators/u1-hmc/out_L=16.h5', help="The file with its directory containing the input .h5 file with the gauge links configurations each as a (L,L,2) array")
@click.option("--output_directory", "output_directory", "-out_dir", default='/nvme/h/cy22sg1/Improved_Wilson_operators/Critical_mass_analysis/processing/to_be_processed/', help="The directory that will contain the output .h5 files with the time-dependent pion correlator arrays")
@click.option("--bare_mass_value", "-mb", "bare_mass_value", default=0.0, help="The bare mass value of the fermion action")
@click.option("--derivative_index", "-d_idx", "derivative_stencil_index", default=0, help="0: Standard, 1: Brillouin, 2: Isotropic")
@click.option("--laplacian_index", "-l_idx", "laplacian_stencil_index", default=0, help="0: Standard, 1: Tilted, 2: Brillouin, 3: Isotropic")
@click.option("--initial_configuration_index", "-in_idx", "initial_gauge_links_configuration_index", default=20, help="Index for the chosen gauge links configuration dataset.")
@click.option("--precision", "-CG_p", "CG_precision", default=1E-7, help="Precision for the CG algorithm")
@click.option("--overlap_boolean", "-op_bn", "overlap_operator_boolean", default=False, help="Choose whether to calculate the improved Wilson or the corresponding overlap operator")
@click.option("--sign_function_boolean", "-sg_bn", "sign_function_boolean", default=True, help="Choose whether to use Chebyshev polynomials (True) or Kenney-Laub iterates for constructing the matrix sign function (False).")
@click.option("-NSF", "number_of_SF_iterations", default=70, help="Sign function iterations")
@click.option("--KL_iterations", "-KL_iter", "KL_iterations", default=1, help="Number of iterations of the KL iterates.")

def main(input_file, output_directory, bare_mass_value, derivative_stencil_index, laplacian_stencil_index, initial_gauge_links_configuration_index, CG_precision, number_of_SF_iterations, overlap_operator_boolean, sign_function_boolean, KL_iterations):

    # Global constants      
    FERMION_DIMENSIONS = 4
    SKIP_CONFIGURATIONS = 20

    world_comm = MPI.COMM_WORLD
    world_rank = world_comm.Get_rank()
    world_size = world_comm.Get_size()

    assert (world_size%FERMION_DIMENSIONS == 0), f'Number of precesses must be a multiple of {FERMION_DIMENSIONS}.'
    '''
    TODO: Check again.
    '''

    # Create a new communicator to group continuous ranks into groups of FERMION_DIMENSIONS
    new_comm = world_comm.Split(color=world_rank//FERMION_DIMENSIONS)
    new_rank = new_comm.Get_rank()
    new_size = new_comm.Get_size()

    # Create a new communicator to group every FERMION_DIMENSIONSth rank together
    new_new_comm = world_comm.Split(color=world_rank%FERMION_DIMENSIONS)
    new_new_rank = new_new_comm.Get_rank()
    new_new_size = new_new_comm.Get_size()

    if (world_rank == 0):
        '''Open the input H5DF file, extract the list with the dataset directories, and construct a global lattice size structure object.'''

        start = time.time()

        assert os.path.isfile(input_file), 'Input H5DF file does not exist.'

        input_HDF5_file = h5py.File(input_file, 'r')
        '''
        TODO: Check if it's an H5DF file indeed with try statement.
        '''

        # Extract the list with the dataset directories
        input_HDF5_file_datasets_directories_list = list()
        def extracting_datasets_names(subgroup_name):
            if '/u' in subgroup_name:
                input_HDF5_file_datasets_directories_list.append(subgroup_name)
        input_HDF5_file.visit(extracting_datasets_names)

        assert len(input_HDF5_file_datasets_directories_list) != 0, 'Given input H5DF file is empty.'
        number_of_available_gauge_configurations = len(input_HDF5_file_datasets_directories_list)

        assert isinstance(initial_gauge_links_configuration_index, int) and (initial_gauge_links_configuration_index >= 0) and (initial_gauge_links_configuration_index < number_of_available_gauge_configurations), f'The "initial_gauge_links_configuration_index" parameter must be a non-negative integer smaller than {number_of_available_gauge_configurations}.'

        # Extract the gauge links field corresponding to the passed initial gauge links configuration index
        gauge_links_field_array = np.array(input_HDF5_file[input_HDF5_file_datasets_directories_list[initial_gauge_links_configuration_index]])
        # The global lattice structure is configured using the gauge links field GaugeLinksField class constructor
        U1_gauge_links_field = gauge.GaugeLinksField.from_gauge_links_field_array(gauge_links_field_array)
        # Pass the gauge links field array
        U1_gauge_links_field.gauge_links_field_array = gauge_links_field_array

        # Extract the fundamental parameters and denote tem using convenient abbreviations to be passed to all the processes
        L = U1_gauge_links_field.lattice_size
        d = U1_gauge_links_field.lattice_dimensions
        f = U1_gauge_links_field.fermion_dimensions
        u = U1_gauge_links_field.theory_label

        # The global lattice structure is not useful anymore
        del U1_gauge_links_field

    else:
        input_HDF5_file_datasets_directories_list = None
        L, d, f, u = (None, None, None, None)

    input_HDF5_file_datasets_directories_list = new_new_comm.bcast(input_HDF5_file_datasets_directories_list, root=0)
    
    L, d, f, u = world_comm.bcast((L, d, f, u), root=0)

    # Initializing the final output array for all processes
    time_dependent_pion_correlator_per_configuration_2D_array = np.zeros((new_new_size, L))

    if (new_rank == 0):
        '''Every 4th process must be assigned a distinct gauge links field configuration that will broadcast to the rest of the ranks of the new_comm.'''

        input_HDF5_file = h5py.File(input_file, 'r')

        gauge_links_configuration_index = initial_gauge_links_configuration_index + new_new_rank*SKIP_CONFIGURATIONS

        # Extract the corresponding gauge links field to the passed index
        gauge_links_field_array = np.array(input_HDF5_file[input_HDF5_file_datasets_directories_list[gauge_links_configuration_index]])

        # Partial unit array to be used to evaluate the partial quark propagator
        psi_vector_field_array = np.eye(4, f*L**d, dtype=np.complex_)

        # TODO:
        inverse_of_operator_vector_array = np.zeros_like(psi_vector_field_array)

    else:
        psi_vector_field_array = None
        gauge_links_field_array = None
        inverse_of_operator_vector_array = None

    # Parts of the psi_vector_field_array are passed to each new_comm rank as appropriate unit vectors
    psi_vector_field = np.zeros((f*L**d,), dtype=np.complex_)
    new_comm.Scatterv([psi_vector_field_array, MPI.COMPLEX], [psi_vector_field, MPI.COMPLEX], root=0)

    # The same exact gauge_links_field_array is passed to each new_comm rank
    gauge_links_field_array = new_comm.bcast(gauge_links_field_array, root=0)

    # A local lattice and gauge structures are constructed
    U1_gauge_links_field = gauge.GaugeLinksField.from_gauge_links_field_array(gauge_links_field_array)
    # The gauge links field array is passed to the local gauge object
    U1_gauge_links_field.gauge_links_field_array = gauge_links_field_array
    # Construct the local improved Wilson operator object using the local gauge object
    improved_Wilson_operator = operators.ImprovedWilsonOperator(
        U1_gauge_links_field,
        derivative_stencil_index,
        laplacian_stencil_index,
        bare_mass_value
    )

    # MAIN CALCULATION
    # Calculate the column of the quark propagator corresponding to the gauge field and unit vector passed to the rank
    inverse_of_operator_vector = np.zeros_like(psi_vector_field)
    x0 = np.zeros((f*(L**d),), dtype=np.complex_)
    precision_info=f'_CG={CG_precision:.0e}'
    if not overlap_operator_boolean:
        operator_type_label = 'Improved_Wilson_operator'

        # quark propagator column
        inverse_of_operator_vector = improved_Wilson_operator.improved_Wilson_operator_functional_conjugate_gradient_solution_function(psi_vector_field, x0, CG_precision)
    
    else:
        if sign_function_boolean:
            sign_function_label = 'Chebyshev_'
            precision_info += f'_NSF={number_of_SF_iterations}'
        else:
            'KL1_'
            precision_info += f'_KL_iter={KL_iterations}'

        operator_type_label = sign_function_label+'overlap_operator'

        # Construct the local overlap operator object using the local improved Wilson operator
        overlap_operator = operators.OverlapOperator(improved_Wilson_operator)

        if (new_rank == 0):
            '''Every 4th process must calculate the corresponding kernel eigenvalues and broadcast them to the rest of the .'''

            # Calculate the eigenvalues of the square of the kernel
            class ExplicitLinearOperatorClass(LinearOperator):
                def __init__(self, N, dtype='float32'):
                    self.N = N
                    self.shape = (self.N, self.N)
                    self.dtype = np.dtype(dtype)
                def kernel_function(self, x):
                    return overlap_operator.improved_Wilson_operator_linear_function(x) - x
                def _matvec(self, x):
                    return overlap_operator.gamma5_function(self.kernel_function(overlap_operator.gamma5_function(self.kernel_function(x))))

            N = overlap_operator.fermion_dimensions*(overlap_operator.lattice_size**overlap_operator.lattice_dimensions)

            linear_operator_object = ExplicitLinearOperatorClass(N, dtype=np.complex_)

            largest_eigenvalue = eigsh(linear_operator_object, k=1, which = 'LM', tol=1E-5)[0]
            smallest_eigenvalue = eigsh(linear_operator_object, k=1, which = 'SM', tol=1E-5)[0]

            # alpha, beta = np.sqrt(smallest_eigenvalue[0], largest_eigenvalue[0])
            alpha, beta = np.sqrt(smallest_eigenvalue[0]), np.sqrt(largest_eigenvalue[0])
        
        else:
            alpha, beta = (None, None)
        
        alpha, beta = new_comm.bcast((alpha, beta), root=0)

        # quark propagator column
        inverse_of_operator_vector = overlap_operator.overlap_operator_functional_conjugate_gradient_solution_function(alpha, beta, psi_vector_field, x0, CG_precision, N=number_of_SF_iterations)

    # Pass the column of the quark propagator to the new_comm root
    new_comm.Gatherv([inverse_of_operator_vector, MPI.COMPLEX], [inverse_of_operator_vector_array, MPI.COMPLEX], root=0)

    if (new_rank == 0):
        '''Use the assembled partial quark propagator to calculate the time-dependent pion correlator which is to be passed to the world_comm root.'''

        time_dependent_pion_correlator_per_configuration_array = np.real(improved_Wilson_operator.time_dependent_pion_correlator_function(np.transpose(inverse_of_operator_vector_array)))

        time_dependent_pion_correlator_per_configuration_array = np.array(time_dependent_pion_correlator_per_configuration_array)

    else:
        time_dependent_pion_correlator_per_configuration_array = np.zeros((L,))

    # Gather all pion correlator arrays from new_comm root to world_comm root
    new_new_comm.Gatherv(time_dependent_pion_correlator_per_configuration_array,time_dependent_pion_correlator_per_configuration_2D_array, root=0)

    if (world_rank == 0):
        '''Write the output array to a binary file with the appropriate filename.'''

        # Construct the appropriate filename
        output_filename = operator_type_label+'_'+constants.laplacian_stencils_labels_array[laplacian_stencil_index]+'_laplacian_'+constants.derivative_stencils_labels_array[derivative_stencil_index]+'_derivative'+f'_L={improved_Wilson_operator.lattice_size}'+'_mb={:.2f}'.format(bare_mass_value)+precision_info+'_configs_total='+str(new_new_size)+'_initial_config='+str(initial_gauge_links_configuration_index)

        # Pass output array to binary file
        time_dependent_pion_correlator_per_configuration_2D_array.tofile(output_directory+output_filename)

        # Calculate elapsed time in seconds
        end = time.time()

        with open("./runtimes.txt", "a") as auxiliary_file:
            auxiliary_file.write((datetime.now()).strftime("%d/%m/%Y %H:%M:%S")+' '+output_filename+' '+f'Elapsed time: {end - start:.2f}\n')

        print("Elapsed time", end - start)

        print("Done!")


if __name__ == "__main__":
    main()