'''Parallel calculation of the time-dependence of the Pion correlator'''
import numpy as np
from scipy.sparse.linalg import eigs, eigsh
from scipy.sparse.linalg import LinearOperator
import h5py
import click
import sys
import os.path
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
import time

sys.path.append('..')
import custom_library.auxiliary as auxiliary
import custom_library.gauge as gauge
import custom_library.operators as operators
import custom_library.constants as constants
from mpi4py import MPI


@click.command()
@click.option("--input_file", "input_file", "-in_file", default='../../u1-hmc/out_L=32.h5', help="The file with its directory containing the input .h5 file with the gauge links configurations each as a (L,L,2) array")
@click.option("--bare_mass_value", "-mb", "bare_mass_value", default=0.0, help="The bare mass value of the fermion action")
@click.option("--derivative_index", "-d_idx", "derivative_stencil_index", default=0, help="0: Standard, 1: Brillouin, 2: Isotropic")
@click.option("--laplacian_index", "-l_idx", "laplacian_stencil_index", default=0, help="0: Standard, 1: Tilted, 2: Brillouin, 3: Isotropic")
@click.option("--configuration_index", "-idx", "gauge_links_configuration_index", default=20, help="Index for the chosen gauge links configuration dataset.")
@click.option("--column_index", "-c_idx", "column_index", default=0, help="Index for the column of the quark propagator.")
@click.option("--precision", "-CG_p", "CG_precision", default=1E-7, help="Precision for the CG algorithm")
@click.option("-NSF", "number_of_SF_iterations", default=50, help="Sign function iterations")
@click.option("--output_directory", "output_directory", "-out_dir", default='./to_be_processed/', help="The directory that will contain the output .h5 files with the time-dependent pion correlator arrays")
@click.option("--overlap_boolean", "-op_bn", "overlap_operator_boolean", default=True, help="choose whether to calculate the improved Wilson or the corresponding overlap operator")

def main(input_file, output_directory, bare_mass_value, derivative_stencil_index, laplacian_stencil_index, gauge_links_configuration_index, column_index, CG_precision, number_of_SF_iterations, overlap_operator_boolean):

	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()

	if (rank == 0):

		start = time.time()
		
		fermion_dimensions = 4
		TESTING = False

		assert size >= fermion_dimensions, f'Number of precesses must be greater than {fermion_dimensions}.'
          
		assert os.path.isfile(input_file), 'Given input H5DF file does not exist.'

		input_HDF5_file = h5py.File(input_file, 'r')
		'''
		TODO: Check if it's an H5DF file indeed with try statement.
		'''
		input_HDF5_file_datasets_directories_list = list()
		def extracting_datasets_names(subgroup_name):
			if '/u' in subgroup_name:
				input_HDF5_file_datasets_directories_list.append(subgroup_name)
		input_HDF5_file.visit(extracting_datasets_names)

		assert len(input_HDF5_file_datasets_directories_list) != 0, 'Given input H5DF file is empty.'
		number_of_available_gauge_configurations = len(input_HDF5_file_datasets_directories_list)

		assert isinstance(gauge_links_configuration_index, int) and (gauge_links_configuration_index >= 0) and (gauge_links_configuration_index < number_of_available_gauge_configurations), f'The "gauge_links_configuration_index" parameter must be a non-negative integer less than {number_of_available_gauge_configurations}.'

		# Extract the corresponding gauge links field to the passed index
		gauge_links_field_array = np.array(input_HDF5_file[input_HDF5_file_datasets_directories_list[gauge_links_configuration_index]])
		# The lattice structure is configured using the gauge links field
		U1_gauge_links_field = gauge.GaugeLinksField.from_gauge_links_field_array(gauge_links_field_array)

		# Pass the gauge links field array
		U1_gauge_links_field.gauge_links_field_array = gauge_links_field_array

		print(U1_gauge_links_field.lattice_size, size)

		assert U1_gauge_links_field.lattice_size%size == 0, 'Number of processes must divide the lattice.'

		# Use convenient abbreviation for the fundamental parameters to be broadcasted to all the processes
		L = U1_gauge_links_field.lattice_size
		d = U1_gauge_links_field.lattice_dimensions
		f = U1_gauge_links_field.fermion_dimensions
		u = U1_gauge_links_field.theory_label

		psi_vector_field = np.zeros((f*(L**d),), dtype=np.complex_)
		psi_vector_field[column_index] = 1

		# partial_psi_vector_field_array = np.eye(f,f*(L**d), dtype=np.complex_)
		# print(np.shape(partial_psi_vector_field_array))
		
		#  Construct a global improved Wilson operator object for testing purposes
		improved_Wilson_operator_vector = np.zeros(np.shape(psi_vector_field), dtype=np.complex_)
		improved_Wilson_operator = operators.ImprovedWilsonOperator(
			U1_gauge_links_field,
			derivative_stencil_index,
			laplacian_stencil_index,
			bare_mass_value
		)

		if not overlap_operator_boolean:
			operator_type_label = 'Improved_Wilson_operator'

			inverse_of_improved_Wilson_operator_vector = np.zeros(np.shape(psi_vector_field), dtype=np.complex_)

		else:
			overlap_operator = operators.OverlapOperator(improved_Wilson_operator)

			operator_type_label = 'Chebyshev_overlap_operator'

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

			alpha = np.sqrt(smallest_eigenvalue[0])
			beta = np.sqrt(largest_eigenvalue[0])

		if TESTING:

			test_improved_Wilson_operator_vector = improved_Wilson_operator.improved_Wilson_operator_linear_function(psi_vector_field)

			x0 = np.ones((f*(L**d),), dtype=np.complex_)
			test_inverse_of_improved_Wilson_operator_vector = improved_Wilson_operator.improved_Wilson_operator_functional_conjugate_gradient_solution_function(psi_vector_field, x0)

			print(test_inverse_of_improved_Wilson_operator_vector)
		
	else:
		gauge_links_field_array = None
		psi_vector_field = None
		improved_Wilson_operator_vector = None
		inverse_of_improved_Wilson_operator_vector = None
		L, d, f, u = (None, None, None, None)
		alpha, beta = (None, None)
		local_inverse_of_improved_Wilson_operator_vector = None

	# Broadcast all the lattice and gauge structures info
	L, d, f, u = comm.bcast((L, d, f, u), root=0)
	alpha, beta = comm.bcast((alpha, beta), root=0)

	# Distribute the gauge links field to the processes
	local_gauge_links_field_array = np.zeros((L//size, L, d), dtype=np.complex_)
	comm.Scatterv([gauge_links_field_array, MPI.COMPLEX], [local_gauge_links_field_array, MPI.COMPLEX], root=0)

	# Append a halo to the local gauge links field
	extended_local_gauge_links_field_array = np.zeros((L//size+2, L, 2), dtype=np.complex_)
	extended_local_gauge_links_field_array[1:L//size+1,:] = local_gauge_links_field_array
	# Upper halo
	comm.Sendrecv(local_gauge_links_field_array[-1,:], dest=(rank+1)%size, sendtag=(rank+1)%size, recvbuf=extended_local_gauge_links_field_array[0,:], source=(rank-1)%size, recvtag=rank)
	# Lower halo
	comm.Sendrecv(local_gauge_links_field_array[0,:], dest=(rank-1)%size, sendtag=(rank-1)%size, recvbuf=extended_local_gauge_links_field_array[-1,:], source=(rank+1)%size, recvtag=rank)

	# Reconstruct the gauge field object as local
	local_U1_gauge_links_field = gauge.GaugeLinksField(
		lattice_size=L,
		lattice_dimensions=d,
		fermion_dimensions=f,
		theory_label=u
	)
	local_U1_gauge_links_field.gauge_links_field_array = extended_local_gauge_links_field_array

	# Construct the local improved Wilson operator object
	local_improved_Wilson_operator = operators.ImprovedWilsonOperator(
		local_U1_gauge_links_field,
		derivative_stencil_index,
		laplacian_stencil_index,
		bare_mass_value
	)

	# Distribute the initial vector field to the processes
	local_psi_vector_field = np.zeros((f*L**d//size,), dtype=np.complex_)
	extended_local_psi_vector_field = np.zeros((f*(L//size+2)*L,), dtype=np.complex_)
	comm.Scatterv(psi_vector_field, local_psi_vector_field, root=0)

	# Append a halo to the initial vector field
	extended_local_psi_vector_field[f*L:f*(L//size+1)*L] = local_psi_vector_field
	# Lower element addition
	comm.Sendrecv(local_psi_vector_field[f*(L//size-1)*L:], dest=(rank+1)%size, sendtag=(rank+1)%size, recvbuf=extended_local_psi_vector_field[:f*L], source=(rank-1)%size, recvtag=rank)
	# Upper element addition
	comm.Sendrecv(np.array(local_psi_vector_field[:f*L]), dest=(rank-1)%size, sendtag=(rank-1)%size, recvbuf=extended_local_psi_vector_field[f*(L//size+1)*L:], source=(rank+1)%size, recvtag=rank)

	local_x0 = np.ones(np.shape(extended_local_psi_vector_field), dtype=np.complex_)
	if not overlap_operator_boolean:
		# Calculate the column of the column improved Wilson operator
		local_operator_vector = local_improved_Wilson_operator.parallel_improved_Wilson_operator_linear_function(extended_local_psi_vector_field)

		local_inverse_of_improved_Wilson_operator_vector = local_improved_Wilson_operator.parallel_improved_Wilson_operator_functional_conjugate_gradient_solution_function(extended_local_psi_vector_field, local_x0, CG_precision)

	else:
		local_overlap_operator = operators.OverlapOperator(local_improved_Wilson_operator)

		# Calculate the column of the column overlap operator
		local_operator_vector = local_overlap_operator.parallel_massive_overlap_operator_linear_function(alpha, beta, extended_local_psi_vector_field)

	# Gather all pieces together to root 0
	comm.Gatherv([local_operator_vector[f*L:f*(L//size+1)*L], MPI.COMPLEX], [improved_Wilson_operator_vector, MPI.COMPLEX], root=0)
	# comm.Gatherv([local_inverse_of_improved_Wilson_operator_vector[f*L:f*(L//size+1)*L], MPI.COMPLEX], [inverse_of_improved_Wilson_operator_vector, MPI.COMPLEX], root=0)

	if (rank == 0):

		print(np.shape(improved_Wilson_operator_vector))

		if TESTING:

			auxiliary.compare_two_matrices((improved_Wilson_operator_vector), (test_improved_Wilson_operator_vector))

			auxiliary.compare_two_matrices(inverse_of_improved_Wilson_operator_vector, test_inverse_of_improved_Wilson_operator_vector)

		output_filename = operator_type_label+'_'+constants.laplacian_stencils_labels_array[laplacian_stencil_index]+'_laplacian_'+constants.derivative_stencils_labels_array[derivative_stencil_index]+'_derivative'+f'_L={L}'+'_mb={:.2f}'.format(bare_mass_value)+'_config_index='+str(gauge_links_configuration_index)+'_column_index='+str(column_index)

		# inverse_of_improved_Wilson_operator_vector.tofile(output_directory+output_filename)

		print(output_filename)
		print(np.shape(improved_Wilson_operator_vector))
		print(sys.getsizeof(improved_Wilson_operator_vector)) # in bytes
		end = time.time() # in seconds
		print(end - start)
		print()
		
		print("Done!")


if __name__ == "__main__":
    main()
