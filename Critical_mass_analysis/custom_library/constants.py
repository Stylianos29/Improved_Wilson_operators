import numpy as np

# __all__ = []

SU2_dof = 4
SU3_dof = 8

# gamma matrices in the chiral representation
gamma_matrices = np.array([
  [[0, 0, 0, +0-1.j], [0, 0, +0-1.j, 0], [0, +0+1.j, 0, 0], [+0+1.j, 0, 0, 0]],
  [[0, 0, 0, -1+0.j], [0, 0, +1+0.j, 0], [0, +1+0.j, 0, 0], [-1+0.j, 0, 0, 0]],
  [[0, 0, +0-1.j, 0], [0, 0, 0, +0+1.j], [+0+1.j, 0, 0, 0], [0, +0-1.j, 0, 0]],
  [[0, 0, +1+0.j, 0], [0, 0, 0, +1+0.j], [+1+0.j, 0, 0, 0], [0, +1+0.j, 0, 0]],
  [[+1+0.j, 0, 0, 0], [0, +1+0.j, 0, 0], [0, 0, -1+0.j, 0], [0, 0, 0, -1+0.j]]
])

# SU3 lambda matrices
lambda_matrices = np.array([
np.array([ [ 0, 1., 0], [1., 0, 0], [0, 0, 0] ]),
np.array([ [ 0, -1.j, 0], [1.j, 0, 0], [0, 0, 0] ]),
np.array([ [ 1., 0, 0], [0, -1., 0], [0, 0, 0] ]),
np.array([ [ 0, 0, 1], [0, 0, 0], [1., 0, 0] ]),
np.array([ [ 0, 0, -1.j], [0, 0, 0], [1.j, 0, 0] ]),
np.array([ [ 0, 0, 0], [0, 0, 1], [0, 1., 0] ]),
np.array([ [ 0, 0, 0], [0, 0, -1.j], [0, 1.j, 0] ]),
(1/np.sqrt(3))*np.array([ [ 1., 0, 0], [0, 1., 0], [0, 0, -2] ])
])

# Derivative stencils
standard_derivative_stencil = np.array([[0, 0, 0], [-1, 0, +1], [0, 0, 0]])/2.0
brillouin_derivative_stencil = np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]])/8.0
isotropic_derivative_stencil = np.array([[-1, 0, +1], [-4, 0, +4], [-1, 0, +1]])/12.0

x_direction_derivative_stencils_array = [standard_derivative_stencil, brillouin_derivative_stencil, isotropic_derivative_stencil]
y_direction_derivative_stencils_array = [np.transpose(standard_derivative_stencil), np.transpose(brillouin_derivative_stencil), np.transpose(isotropic_derivative_stencil)]
derivative_stencils_array = [x_direction_derivative_stencils_array, y_direction_derivative_stencils_array]
derivative_stencils_labels_array = ['standard', 'brillouin', 'isotropic']

# Laplacian stencils
standard_laplacian_stencil = np.array([[0, +1, 0], [+1, -4, +1], [0, +1, 0]])/1.0
tilted_laplacian_stencil = np.array([[+1, 0, +1], [0, -4, 0], [+1, 0, +1]])/2.0
brillouin_laplacian_stencil = np.array([[+1, +2, +1], [+2, -12, +2], [+1, +2, +1]])/4.0
isotropic_laplacian_stencil = np.array([[+1, +4, +1], [+4, -20, +4], [+1, +4, +1]])/6.0

laplacian_stencils_array = [standard_laplacian_stencil, tilted_laplacian_stencil, brillouin_laplacian_stencil, isotropic_laplacian_stencil]
laplacian_stencils_labels_array = ['standard', 'tilted', 'brillouin', 'isotropic']
