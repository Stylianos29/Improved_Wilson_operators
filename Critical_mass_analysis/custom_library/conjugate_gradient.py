'''
TODO: denote properly public and private members of classes
'''

import numpy as np

def generic_reproducing_inverse_of_matrix_function(generic_matrix_inverse_column_function, parameters, A):

	A = np.array(A)
	assert (A.ndim == 2) and (np.shape(A)[0] == np.shape(A)[1]), 'Input array must be a 2-dimensional square array.'
	matrix_size = np.shape(A)[0]


	# helper function
	counter_array = list()
	def abbreviated_matrix_function(column_vector):
		x0 = np.zeros((matrix_size,), dtype=np.complex_)
		
		result = generic_matrix_inverse_column_function(A, x0, column_vector, *parameters)

		counter_array.append(result[1])

		return result[0]
	
	reproduced_matrix = map(abbreviated_matrix_function, np.identity(matrix_size, dtype=np.complex_))
	reproduced_matrix = np.array(list(reproduced_matrix))

	counter_array = np.array(counter_array)

	return reproduced_matrix.T, counter_array

def inverse_of_hermitian_matrix(A, x0, b, CG_precision=1E-6):
	'''Application of the conjugate gradient algorithm for calculating the inverse of a hermitian matrix.'''

	assert np.all(np.isclose(np.conjugate(np.transpose(A)), A, atol=1e-13)), 'Input matrix must be real symmetric or complex hermitian.'

	b_norm = np.linalg.norm(b)

	r = b - A.dot(x0)
	p = r
	x = x0

	counter = 0
	while True:

		residue_norm = np.linalg.norm(r)
		Ap = A.dot(p)

		alpha_factor = residue_norm**2 / (np.conjugate(p)).dot(Ap)
		
		x = x + alpha_factor*p
		r = r - alpha_factor*Ap

		if (np.linalg.norm(r)/b_norm < CG_precision):
			break

		beta_factor = (np.linalg.norm(r) / residue_norm)**2

		p = r + beta_factor*p

		counter += 1
	
	return x, counter-1
