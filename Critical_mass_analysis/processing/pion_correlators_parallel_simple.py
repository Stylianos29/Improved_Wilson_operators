'''Calculating the time-dependence of the Pion correlator'''

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import LinearOperator
import h5py
import click
import sys
import os.path
import time
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
from datetime import datetime
import psutil

# custom modules
sys.path.append('../')
import custom_library.auxiliary as auxiliary
import custom_library.constants as constants
import custom_library.gauge as gauge
import custom_library.operators as operators

# Global constants
FERMION_DIMENSIONS = 4
LOWER_BOUND_GAUGE_CONFIGURATION_INDEX = 20
GAUGE_CONFIGURATION_INDEX_SKIP = 20
KL_ITERATIONS_RANGE = [1, 2, 3]


@click.command()
@click.option("--input_file", "input_file", "-in_file", default='../../u1-hmc/out_L=16.h5', help="The file with its directory containing the input .h5 file with the gauge links configurations each as a (L,L,2) array")
@click.option("--output_directory", "output_directory", "-out_dir", default='./to_be_processed/', help="The directory that will contain the output .h5 files with the time-dependent pion correlator arrays")
@click.option("--bare_mass_value", "-mb", "bare_mass_value", default=0.00, help="The bare mass value of the fermion action")
@click.option("--derivative_index", "-d_idx", "derivative_stencil_index", default=0, help="0: Standard, 1: Brillouin, 2: Isotropic")
@click.option("--laplacian_index", "-l_idx", "laplacian_stencil_index", default=0, help="0: Standard, 1: Tilted, 2: Brillouin, 3: Isotropic")
@click.option("--precision", "-CG_p", "CG_precision", default=1E-6, help="Precision for the CG algorithm")
@click.option("--initial_configuration_index", "-ic_idx", "initial_gauge_configuration_index", default=20, help="number of initial configurations to skip before the rest are used to calculation")
@click.option("--overlap_boolean", "-op_bn", "overlap_operator_boolean", default=False, help="choose whether to calculate the improved Wilson or the corresponding overlap operator")
@click.option("--sign_function_boolean", "-sg_bn", "sign_function_boolean", default=True, help="Choose whether to use Chebyshev polynomials (True) or Kenney-Laub iterates for constructing the matrix sign function (False).")
@click.option("-NSF", "number_of_SF_iterations", default=50, help="Sign function iterations")
@click.option("--KL_iterations", "-KL_iter", "KL_iterations", default=1, help="Number of iterations of the KL iterates.")
@click.option("--testing_boolean", "-test_bn", "testing_boolean", default=False, help="Set script in testing mode without output apart from certain stats.")

def main(input_file, output_directory, bare_mass_value, derivative_stencil_index, laplacian_stencil_index, CG_precision, initial_gauge_configuration_index, overlap_operator_boolean, sign_function_boolean, number_of_SF_iterations, KL_iterations, testing_boolean):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if (rank == 0):

        start = time.time()

        assert os.path.isfile(input_file), 'Given input H5DF file does not exist.'

        input_HDF5_file = h5py.File(input_file, 'r')
        # Extract the list with the dataset directories
        input_HDF5_file_datasets_directories_list = list()
        def extracting_datasets_names(subgroup_name):
            if '/u' in subgroup_name:
                input_HDF5_file_datasets_directories_list.append(subgroup_name)
        input_HDF5_file.visit(extracting_datasets_names)

        assert len(input_HDF5_file_datasets_directories_list) != 0, 'Given input H5DF file is empty.'
        number_of_available_gauge_configurations = len(input_HDF5_file_datasets_directories_list)

        assert isinstance(initial_gauge_configuration_index, int) and (initial_gauge_configuration_index >= LOWER_BOUND_GAUGE_CONFIGURATION_INDEX) and (initial_gauge_configuration_index < number_of_available_gauge_configurations), f'The "gauge_links_configuration_index" parameter must be a non-negative integer less than {number_of_available_gauge_configurations}.'

        # Extract the lattice size from the gauge links field corresponding to the initial configuration index
        gauge_links_field_array = np.array(input_HDF5_file[input_HDF5_file_datasets_directories_list[initial_gauge_configuration_index]])
        lattice_size = np.shape(gauge_links_field_array)[0]

        # Output array
        time_dependent_pion_correlator_per_configuration_2D_array = np.zeros((size, lattice_size))

    else:
        input_HDF5_file_datasets_directories_list = None
        time_dependent_pion_correlator_per_configuration_2D_array = None

    input_HDF5_file = h5py.File(input_file, 'r')

    input_HDF5_file_datasets_directories_list = comm.bcast(input_HDF5_file_datasets_directories_list, root=0)

    gauge_configuration_index = initial_gauge_configuration_index + rank*GAUGE_CONFIGURATION_INDEX_SKIP

    # The lattice structure is configured using the gauge links field corresponding to the configuration index
    gauge_links_field_array = np.array(input_HDF5_file[input_HDF5_file_datasets_directories_list[gauge_configuration_index]])

    # Pass the gauge links field array for constructing gauge links field object
    U1_gauge_links_field = gauge.GaugeLinksField.from_gauge_links_field_array(gauge_links_field_array)

    # Pass the gauge links field array for constructing gauge links field object
    U1_gauge_links_field.gauge_links_field_array = gauge_links_field_array

    # Construct the improved Wilson operator object
    improved_Wilson_operator = operators.ImprovedWilsonOperator(U1_gauge_links_field, derivative_stencil_index, laplacian_stencil_index, bare_mass_value)

    # initializing the precision info label
    precision_info = f'CG={CG_precision:.0e}'

    if not overlap_operator_boolean:
        operator_type_label = 'Improved_Wilson_operator'

        # Calculate the partial quark propagator
        partial_inverse_operator = improved_Wilson_operator.inverse_Wilson_operator_function(precision=CG_precision)

        # Time-dependent Pion correlator
        time_dependent_pion_correlator_per_configuration_array = np.real(improved_Wilson_operator.time_dependent_pion_correlator_function(partial_inverse_operator))

    else:
        # Construct the overlap operator object
        overlap_operator = operators.OverlapOperator(improved_Wilson_operator)

        if sign_function_boolean:
            operator_type_label = 'Chebyshev_overlap_operator'
            
            # Add the number of Chebyshev terms to the precision info label
            assert isinstance(number_of_SF_iterations, int) and number_of_SF_iterations > 0, 'The number of Chebyshev terms must be a positive integer.'
            precision_info += f'_NSF={number_of_SF_iterations}'

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

            N = FERMION_DIMENSIONS*(overlap_operator.lattice_size**overlap_operator.lattice_dimensions)

            linear_operator_object = ExplicitLinearOperatorClass(N, dtype=np.complex_)

            largest_eigenvalue = eigsh(linear_operator_object, k=1, which = 'LM', tol=1E-5)[0]
            smallest_eigenvalue = eigsh(linear_operator_object, k=1, which = 'SM', tol=1E-5)[0]

            alpha = np.sqrt(smallest_eigenvalue[0])
            beta = np.sqrt(largest_eigenvalue[0])

            # Calculate the partial quark propagator
            partial_inverse_operator = overlap_operator.inverse_overlap_operator_function(alpha, beta)

        else:
            assert isinstance(KL_iterations, int) and KL_iterations in KL_ITERATIONS_RANGE, f'The number of KL iterations must be a positive integer in the rage {KL_ITERATIONS_RANGE}.'
            # Add the KL iteration number to the precision info label
            operator_type_label = f'KL{KL_iterations}_overlap_operator'
            precision_info += f'_KL_iter={KL_iterations}'

            if (KL_iterations == 1):
                partial_inverse_operator = overlap_operator.inverse_Kenney_Laub_iterate_overlap_operator_function(CG_precision)
            elif (KL_iterations == 2):
                partial_inverse_operator = overlap_operator.inverse_Kenney_Laub_2nd_iterate_overlap_operator_function(CG_precision)
            elif (KL_iterations == 3):
                partial_inverse_operator = overlap_operator.inverse_Kenney_Laub_3rd_iterate_overlap_operator_function(CG_precision)
        
        # Time-dependent Pion correlator
        time_dependent_pion_correlator_per_configuration_array = np.real(overlap_operator.time_dependent_pion_correlator_function(partial_inverse_operator))

    time_dependent_pion_correlator_per_configuration_array = np.array(time_dependent_pion_correlator_per_configuration_array)

    '''
    TODO: There must be a common .time_dependent_pion_correlator_function method for both ImprovedWilsonOperator and ImprovedWilsonOperator classes.
    '''

    comm.Gatherv(time_dependent_pion_correlator_per_configuration_array, time_dependent_pion_correlator_per_configuration_2D_array, root=0)

    if (rank == 0):

        time_dependent_pion_correlator_per_configuration_2D_array = np.array(time_dependent_pion_correlator_per_configuration_2D_array)

        output_subdirectories = operator_type_label+'/'+constants.laplacian_stencils_labels_array[laplacian_stencil_index]+'_laplacian_'+constants.derivative_stencils_labels_array[derivative_stencil_index]+'_derivative/'+f'L={improved_Wilson_operator.lattice_size}/'+precision_info+'/'

        auxiliary.creating_directory_function(output_directory+'/'+output_subdirectories)

        output_filename = f'mb={bare_mass_value:.2f}_configs_total={size}_initial_config={initial_gauge_configuration_index}'

        if testing_boolean:
            end = time.time()
            print("Number of processes:", size, "Lattice size:", lattice_size,"Elapsed time:", end - start, "Resident memory:", psutil.Process().memory_info().rss / (1024 * 1024), "MB", )

        else:
            # Pass to binary file
            time_dependent_pion_correlator_per_configuration_2D_array.tofile(output_directory+output_subdirectories+output_filename)
            
            end = time.time()
            # Store the elapsed time in a text file
            with open("./runtimes.txt", "a") as auxiliary_file:
                auxiliary_file.write((datetime.now()).strftime("%d/%m/%Y %H:%M:%S")+' '+output_subdirectories.replace("/","_")+output_filename+' '+f'Elapsed time: {end - start:.2f}\n')
            print("Done!")

        
if __name__ == "__main__":
    main()
