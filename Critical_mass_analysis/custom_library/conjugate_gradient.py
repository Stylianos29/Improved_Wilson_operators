'''
TODO: Write description
This module contains 
'''

from typing import Any
import numpy as np
import sys

class InverseOfMatrix:

	def __init__(self, matrix, CG_precision=1e-06, x0=None, *, relative_tolerance=1e-05, absolute_tolerance=1e-08):

		self.matrix = matrix

		self.CG_precision = CG_precision
		
		self.x0 = x0

		assert isinstance(relative_tolerance, float) and (relative_tolerance > 0) and (relative_tolerance < 1), 'The relative tolerance parameter of the CG algorithm must a positive real number smaller than 1.'
		self.relative_tolerance = relative_tolerance
		assert isinstance(absolute_tolerance, float) and (absolute_tolerance > 0) and (absolute_tolerance < 1), 'The absolute tolerance parameter of the CG algorithm must a positive real number smaller than 1.'
		self.absolute_tolerance = absolute_tolerance

		# Assess the type of the input matrix
		self.matrix_label = 'real'
		if np.iscomplexobj(self.matrix):
			self.matrix_label = 'complex'
		if np.allclose(self.matrix, np.transpose(np.conjugate(self.matrix)), rtol=self.relative_tolerance, atol=self.absolute_tolerance):
			if (self.matrix_label == 'real'):
				self.matrix_label = 'real symmetric'
			else:
				self.matrix_label = 'complex Hermitian'
		else:
			if (self.matrix_label == 'real'):
				self.matrix_label = 'real non-symmetric'
			else:
				self.matrix_label = 'complex non-Hermitian'

	def __call__(self, matrix=None, CG_precision=None, x0=None):

		if (matrix is not None):
			self.matrix = matrix
		
		if (CG_precision is not None):
			self.CG_precision = CG_precision

		# ._x0 retains its value unless a new x0 array passed or new input matrix has a different side length than the precious one
		if (x0 is not None):
			self.x0 = x0
		elif (len(self._x0) != self._matrix_side_length):
			self.x0 = None

		# Initialize the list that stores the number of iterations for each application of the CG algorithm
		self.number_of_iterations_list = list()

		return self.inverse_of_matrix_function()

	@property
	def matrix(self):
		return self._matrix

	@matrix.setter
	def matrix(self, matrix):
		matrix = np.array(matrix)
		assert (matrix.ndim == 2) and (np.shape(matrix)[0] == np.shape(matrix)[1]), 'Input array must be a 2-dimensional square array.'
		self._matrix = np.array(matrix)
		
		# Extract the side size of the input matrix
		self._matrix_side_length = np.shape(matrix)[0]

	@property
	def CG_precision(self):
		return self._CG_precision

	@CG_precision.setter
	def CG_precision(self, CG_precision):
		assert isinstance(CG_precision, float) and (CG_precision > 0) and (CG_precision < 1), 'The CG algorithm precision parameter must a positive real number smaller than 1.'
		self._CG_precision = CG_precision

	@property
	def x0(self):
		return self._x0

	@x0.setter
	def x0(self, x0):
		if (x0 is None):
			# Arbitrary choice of an initial x0 column with the same length as the side length of the input matrix
			self.x0 = np.zeros((self._matrix_side_length,), dtype=np.complex_)
		else:
			x0 = np.array(x0)
			assert np.shape(x0) == (self._matrix_side_length,), 'The initial x0 column array must have the size as the side of the 2d input matrix.'
			self._x0 = x0

	def __repr__(self):
		return f'{type(self).__name__}(matrix = ..., CG_precision={self.CG_precision}, x0=..., relative_tolerance={self.relative_tolerance}, absolute_tolerance={self.absolute_tolerance})'

	def __str__(self) -> str:
		return f'\n* This is an instance of the {type(self).__name__} class.\n* A {self.matrix_label} input matrix of side length {self._matrix_side_length} has been pass as an argument.\n* The CG algorithm precision parameter has been set to {self.CG_precision:.1e}.\n* Additionally the absolute and relative tolerance parameters have values {self.absolute_tolerance:.1e} and {self.relative_tolerance:.1e} correspondingly.\n'

	def inverse_of_matrix_function(self):

		if (self.matrix_label == 'real symmetric') or (self.matrix_label == 'complex Hermitian'):
			inverse_of_matrix_function = self.column_inverse_of_hermitian_matrix_function
		else:
			inverse_of_matrix_function = self.column_inverse_of_non_hermitian_matrix_function
		
		reproduced_inverse_of_matrix = map(inverse_of_matrix_function, np.identity(self._matrix_side_length))
		reproduced_inverse_of_matrix = np.array(list(reproduced_inverse_of_matrix))

		return np.transpose(reproduced_inverse_of_matrix)

	def column_inverse_of_hermitian_matrix_function(self, b):
		'''Application of the conjugate gradient algorithm for calculating the inverse of a hermitian matrix.'''

		b_norm = np.linalg.norm(b)

		r = b - (self.matrix).dot(self._x0)
		p = r
		x = self._x0

		# Initialize the iterations counter
		self.number_of_iterations = 0
		while True:

			residue_norm = np.linalg.norm(r)

			Ap = (self.matrix).dot(p)

			alpha_factor = residue_norm**2 / (np.conjugate(p)).dot(Ap)
			
			x = x + alpha_factor*p
			r = r - alpha_factor*Ap

			self.number_of_iterations += 1
			
			if (np.linalg.norm(r)/b_norm < self.CG_precision):
				break

			beta_factor = (np.linalg.norm(r) / residue_norm)**2

			p = r + beta_factor*p
		
		self.number_of_iterations_list.append(self.number_of_iterations)
		
		return x

	def column_inverse_of_non_hermitian_matrix_function(self, b):
		'''Application of the conjugate gradient algorithm for calculating the inverse of a non-hermitian matrix.'''

		b_prime = (np.transpose(np.conjugate(self.matrix))).dot(b)
		b_norm = np.linalg.norm(b_prime)

		square_of_matrix = np.matmul(np.transpose(np.conjugate(self.matrix)), self.matrix)

		r = b_prime - square_of_matrix.dot(self._x0)
		p = r
		x = self._x0

		# Initialize the iterations counter
		self.number_of_iterations = 0
		while True:

			residue_norm = np.linalg.norm(r)

			Ap = square_of_matrix.dot(p)

			alpha_factor = residue_norm**2 / (np.conjugate(p)).dot(Ap)
			
			x = x + alpha_factor*p
			r = r - alpha_factor*Ap

			self.number_of_iterations += 1
			
			if (np.linalg.norm(r)/b_norm < self.CG_precision):
				break

			beta_factor = (np.linalg.norm(r) / residue_norm)**2

			p = r + beta_factor*p

		self.number_of_iterations_list.append(self.number_of_iterations)
		
		return x
