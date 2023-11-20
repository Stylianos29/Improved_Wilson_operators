import numpy as np


# EUCLIDEAN GAMMA MATRICES IN CHIRAL REPRESENTATION
GAMMA_MATRICES = [
    np.array([[0, 0, 0, -1.0j], [0, 0, -1.0j, 0], [0, +1.0j, 0, 0], [+1.0j, 0, 0, 0]]),
    np.array([[0, 0, 0, -1], [0, 0, +1, 0], [0, +1, 0, 0], [-1, 0, 0, 0]]),
    np.array([[0, 0, -1.0j, 0], [0, 0, 0, +1.0j], [+1.0j, 0, 0, 0], [0, -1.0j, 0, 0]]),
    np.array([[0, 0, +1, 0], [0, 0, 0, +1], [+1, 0, 0, 0], [0, +1, 0, 0]]),
    np.array([[+1, 0, 0, 0], [0, +1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]),
]


##############################################
########## GROUP THEORY GENERATORS ###########
##############################################

################### SU(2) ####################

SU2_DOF = 3

# NOTE: The 2x2 identity matrix is included as well for convenience
PAULI_MATRICES = [
    np.array([[+1, 0], [0, +1]]),
    np.array([[0, +1], [+1, 0]]),
    np.array([[0, -1.0j], [+1.0j, 0]]),
    np.array([[+1, 0], [0, -1]]),
]

################### SU(3) ####################

SU3_DOF = 8

# NOTE: The 3x3 identity matrix is included as well for convenience
LAMBDA_MATRICES = [
    np.array([[+1, 0, 0], [0, +1, 0], [0, 0, +1]]),
    np.array([[0, +1, 0], [+1, 0, 0], [0, 0, 0]]),
    np.array([[0, -1.0j, 0], [+1.0j, 0, 0], [0, 0, 0]]),
    np.array([[+1, 0, 0], [0, -1, 0], [0, 0, 0]]),
    np.array([[0, 0, +1], [0, 0, 0], [+1, 0, 0]]),
    np.array([[0, 0, -1.0j], [0, 0, 0], [+1.0j, 0, 0]]),
    np.array([[0, 0, 0], [0, 0, +1], [0, +1, 0]]),
    np.array([[0, 0, 0], [0, 0, -1.0j], [0, +1.0j, 0]]),
    (1 / np.sqrt(3)) * np.array([[+1, 0, 0], [0, +1, 0], [0, 0, -2]]),
]

SU3_STRUCTURE_CONSTANTS_DICTIONARY = {
    (0, 1, 2): 1,
    (0, 2, 1): -1,
    (0, 3, 6): 1 / 2,
    (0, 4, 5): -1 / 2,
    (0, 5, 4): 1 / 2,
    (0, 6, 3): -1 / 2,
    (1, 0, 2): -1,
    (1, 2, 0): 1,
    (1, 3, 5): 1 / 2,
    (1, 4, 6): 1 / 2,
    (1, 5, 3): -1 / 2,
    (1, 6, 4): -1 / 2,
    (2, 0, 1): 1,
    (2, 1, 0): -1,
    (2, 3, 4): 1 / 2,
    (2, 4, 3): -1 / 2,
    (2, 5, 6): -1 / 2,
    (2, 6, 5): 1 / 2,
    (3, 0, 6): -1 / 2,
    (3, 1, 5): -1 / 2,
    (3, 2, 4): -1 / 2,
    (3, 4, 2): 1 / 2,
    (3, 4, 7): np.sqrt(3) / 2,
    (3, 5, 1): 1 / 2,
    (3, 6, 0): 1 / 2,
    (3, 7, 4): -np.sqrt(3) / 2,
    (4, 0, 5): 1 / 2,
    (4, 1, 6): -1 / 2,
    (4, 2, 3): 1 / 2,
    (4, 3, 2): -1 / 2,
    (4, 3, 7): -np.sqrt(3) / 2,
    (4, 5, 0): -1 / 2,
    (4, 6, 1): 1 / 2,
    (4, 7, 3): np.sqrt(3) / 2,
    (5, 0, 4): -1 / 2,
    (5, 1, 3): 1 / 2,
    (5, 2, 6): 1 / 2,
    (5, 3, 1): -1 / 2,
    (5, 4, 0): 1 / 2,
    (5, 6, 2): -1 / 2,
    (5, 6, 7): np.sqrt(3) / 2,
    (5, 7, 6): -np.sqrt(3) / 2,
    (6, 0, 3): 1 / 2,
    (6, 1, 4): 1 / 2,
    (6, 2, 5): -1 / 2,
    (6, 3, 0): -1 / 2,
    (6, 4, 1): -1 / 2,
    (6, 5, 2): 1 / 2,
    (6, 5, 7): -np.sqrt(3) / 2,
    (6, 7, 5): np.sqrt(3) / 2,
    (7, 3, 4): np.sqrt(3) / 2,
    (7, 4, 3): -np.sqrt(3) / 2,
    (7, 5, 6): np.sqrt(3) / 2,
    (7, 6, 5): -np.sqrt(3) / 2,
}


##############################################
###### DERIVATIVE & LAPLACIAN STENCILS #######
##############################################

DERIVATIVE_STENCILS_LABELS = ["standard", "brillouin", "isotropic"]
LAPLACIAN_STENCILS_LABELS = ["standard", "tilted", "brillouin", "isotropic"]

##################### 2D #####################

# 2D LAPLACIAN STENCILS
STANDARD_2D_LAPLACIAN_STENCIL = np.array([[0, +1, 0], [+1, -4, +1], [0, +1, 0]])

TILTED_2D_LAPLACIAN_STENCIL = np.array([[+1, 0, +1], [0, -4, 0], [+1, 0, +1]]) / 2

BRILLOUIN_2D_LAPLACIAN_STENCIL = (
    np.array([[+1, +2, +1], [+2, -12, +2], [+1, +2, +1]]) / 4
)

ISOTROPIC_2D_LAPLACIAN_STENCIL = (
    np.array([[+1, +4, +1], [+4, -20, +4], [+1, +4, +1]]) / 6
)

LAPLACIAN_2D_STENCILS_LIST = [
    STANDARD_2D_LAPLACIAN_STENCIL,
    TILTED_2D_LAPLACIAN_STENCIL,
    BRILLOUIN_2D_LAPLACIAN_STENCIL,
    ISOTROPIC_2D_LAPLACIAN_STENCIL,
]

# 2D DERIVATIVE STENCILS
STANDARD_2D_X_DERIVATIVE_STENCIL = np.array([[0, 0, 0], [-1, 0, +1], [0, 0, 0]]) / 2

BRILLOUIN_2D_X_DERIVATIVE_STENCIL = (
    np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]]) / 8
)

ISOTROPIC_2D_X_DERIVATIVE_STENCIL = (
    np.array([[-1, 0, +1], [-4, 0, +4], [-1, 0, +1]]) / 12
)

DERIVATIVE_2D_X_STENCILS_LIST = [
    STANDARD_2D_X_DERIVATIVE_STENCIL,
    BRILLOUIN_2D_X_DERIVATIVE_STENCIL,
    ISOTROPIC_2D_X_DERIVATIVE_STENCIL,
]

##################### 3D #####################

# 3D LAPLACIANS STENCILS
STANDARD_3D_LAPLACIAN_STENCIL = [
    np.array([[0, 0, 0], [0, +1, 0], [0, 0, 0]]),
    np.array([[0, +1, 0], [+1, -6, +1], [0, +1, 0]]),
    np.array([[0, 0, 0], [0, +1, 0], [0, 0, 0]]),
]

TILTED_3D_LAPLACIAN_STENCIL = [
    np.array([[+1, 0, +1], [0, 0, 0], [+1, 0, +1]]) / 4,
    np.array([[0, 0, 0], [0, -8, 0], [0, 0, 0]]) / 4,
    np.array([[+1, 0, +1], [0, 0, 0], [+1, 0, +1]]) / 4,
]

BRILLOUIN_3D_LAPLACIAN_STENCIL = [
    np.array([[+1, +2, +1], [+2, +4, +2], [+1, +2, +1]]) / 16,
    np.array([[+2, +4, +2], [+4, -56, +4], [+2, +4, +2]]) / 16,
    np.array([[+1, +2, +1], [+2, +4, +2], [+1, +2, +1]]) / 16,
]

ISOTROPIC_3D_LAPLACIAN_STENCIL = [
    np.array([[+1, +6, +1], [+6, +20, +6], [+1, +6, +1]]) / 48,
    np.array([[+6, +20, +6], [+20, -200, +20], [+6, +20, +6]]) / 48,
    np.array([[+1, +6, +1], [+6, +20, +6], [+1, +6, +1]]) / 48,
]

LAPLACIAN_3D_STENCILS_LIST = [
    STANDARD_3D_LAPLACIAN_STENCIL,
    TILTED_3D_LAPLACIAN_STENCIL,
    BRILLOUIN_3D_LAPLACIAN_STENCIL,
    ISOTROPIC_3D_LAPLACIAN_STENCIL,
]

# 3D DERIVATIVES STENCILS
STANDARD_3D_X_DERIVATIVE_STENCIL = [
    np.zeros((3, 3)),
    np.array([[0, 0, 0], [-1, 0, +1], [0, 0, 0]]) / 2,
    np.zeros((3, 3)),
]

BRILLOUIN_3D_X_DERIVATIVE_STENCIL = [
    np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]]) / 32,
    np.array([[-2, 0, +2], [-4, 0, +4], [-2, 0, +2]]) / 32,
    np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]]) / 32,
]

ISOTROPIC_3D_X_DERIVATIVE_STENCIL = [
    np.array([[-1, 0, +1], [-4, 0, +4], [-1, 0, +1]]) / 72,
    np.array([[-4, 0, +4], [-16, 0, +16], [-4, 0, +4]]) / 72,
    np.array([[-1, 0, +1], [-4, 0, +4], [-1, 0, +1]]) / 72,
]

DERIVATIVE_3D_X_STENCILS_LIST = [
    STANDARD_3D_X_DERIVATIVE_STENCIL,
    BRILLOUIN_3D_X_DERIVATIVE_STENCIL,
    ISOTROPIC_3D_X_DERIVATIVE_STENCIL,
]

##################### 4D #####################

# 4D LAPLACIAN STENCILS
STANDARD_4D_LAPLACIAN_STENCIL = [
    [np.zeros((3, 3)), np.array([[0, 0, 0], [0, +1, 0], [0, 0, 0]]), np.zeros((3, 3))],
    [
        np.array([[0, 0, 0], [0, +1, 0], [0, 0, 0]]),
        np.array([[0, +1, 0], [+1, -8, +1], [0, +1, 0]]),
        np.array([[0, 0, 0], [0, +1, 0], [0, 0, 0]]),
    ],
    [np.zeros((3, 3)), np.array([[0, 0, 0], [0, +1, 0], [0, 0, 0]]), np.zeros((3, 3))],
]

TILTED_4D_LAPLACIAN_STENCIL = [
    [
        np.array([[+1, 0, +1], [0, 0, 0], [+1, 0, +1]]) / 8,
        np.zeros((3, 3)),
        np.array([[+1, 0, +1], [0, 0, 0], [+1, 0, +1]]) / 8,
    ],
    [
        np.zeros((3, 3)),
        np.array([[0, 0, 0], [0, -16, 0], [0, 0, 0]]) / 8,
        np.zeros((3, 3)),
    ],
    [
        np.array([[+1, 0, +1], [0, 0, 0], [+1, 0, +1]]) / 8,
        np.zeros((3, 3)),
        np.array([[+1, 0, +1], [0, 0, 0], [+1, 0, +1]]) / 8,
    ],
]

BRILLOUIN_4D_LAPLACIAN_STENCIL = [
    [
        np.array([[+1, +2, +1], [+2, +4, +2], [+1, +2, +1]]) / 64,
        np.array([[+2, +4, +2], [+4, +8, +4], [+2, +4, +2]]) / 64,
        np.array([[+1, +2, +1], [+2, +4, +2], [+1, +2, +1]]) / 64,
    ],
    [
        np.array([[+2, +4, +2], [+4, +8, +4], [+2, +4, +2]]) / 64,
        np.array([[+4, +8, +4], [+8, -240, +8], [+4, +8, +4]]) / 64,
        np.array([[+2, +4, +2], [+4, +8, +4], [+2, +4, +2]]) / 64,
    ],
    [
        np.array([[+1, +2, +1], [+2, +4, +2], [+1, +2, +1]]) / 64,
        np.array([[+2, +4, +2], [+4, +8, +4], [+2, +4, +2]]) / 64,
        np.array([[+1, +2, +1], [+2, +4, +2], [+1, +2, +1]]) / 64,
    ],
]

ISOTROPIC_4D_LAPLACIAN_STENCIL = [
    [
        np.array([[+1, +7, +1], [+7, +40, +7], [+1, +7, +1]]) / 432,
        np.array([[+7, +40, +7], [+40, +100, +40], [+7, +40, +7]]) / 432,
        np.array([[+1, +7, +1], [+7, +40, +7], [+1, +7, +1]]) / 432,
    ],
    [
        np.array([[+7, +40, +7], [+40, +100, +40], [+7, +40, +7]]) / 432,
        np.array([[+40, +100, +40], [+100, -2000, +100], [+40, +100, +40]]) / 432,
        np.array([[+7, +40, +7], [+40, +100, +40], [+7, +40, +7]]) / 432,
    ],
    [
        np.array([[+1, +7, +1], [+7, +40, +7], [+1, +7, +1]]) / 432,
        np.array([[+7, +40, +7], [+40, +100, +40], [+7, +40, +7]]) / 432,
        np.array([[+1, +7, +1], [+7, +40, +7], [+1, +7, +1]]) / 432,
    ],
]

LAPLACIAN_4D_STENCILS_LIST = [
    STANDARD_4D_LAPLACIAN_STENCIL,
    TILTED_4D_LAPLACIAN_STENCIL,
    BRILLOUIN_4D_LAPLACIAN_STENCIL,
    ISOTROPIC_4D_LAPLACIAN_STENCIL,
]

# 4D DERIVATIVES STENCILS
STANDARD_4D_X_DERIVATIVE_STENCIL = [
    [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))],
    [
        np.zeros((3, 3)),
        np.array([[0, 0, 0], [-1, 0, +1], [0, 0, 0]]) / 2,
        np.zeros((3, 3)),
    ],
    [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))],
]

BRILLOUIN_4D_X_DERIVATIVE_STENCIL = [
    [
        np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]]) / 128,
        np.array([[-2, 0, +2], [-4, 0, +4], [-2, 0, +2]]) / 128,
        np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]]) / 128,
    ],
    [
        np.array([[-2, 0, +2], [-4, 0, +4], [-2, 0, +2]]) / 128,
        np.array([[-4, 0, +4], [-8, 0, +8], [-4, 0, +4]]) / 128,
        np.array([[-2, 0, +2], [-4, 0, +4], [-2, 0, +2]]) / 128,
    ],
    [
        np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]]) / 128,
        np.array([[-2, 0, +2], [-4, 0, +4], [-2, 0, +2]]) / 128,
        np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]]) / 128,
    ],
]

ISOTROPIC_4D_X_DERIVATIVE_STENCIL = [
    [
        np.array([[-1, 0, +1], [-4, 0, +4], [-1, 0, +1]]) / 432,
        np.array([[-4, 0, +4], [-16, 0, +16], [-4, 0, +4]]) / 432,
        np.array([[-1, 0, +1], [-4, 0, +4], [-1, 0, +1]]) / 432,
    ],
    [
        np.array([[-4, 0, +4], [-16, 0, +16], [-4, 0, +4]]) / 432,
        np.array([[-16, 0, +16], [-64, 0, +64], [-16, 0, +16]]) / 432,
        np.array([[-4, 0, +4], [-16, 0, +16], [-4, 0, +4]]) / 432,
    ],
    [
        np.array([[-1, 0, +1], [-4, 0, +4], [-1, 0, +1]]) / 432,
        np.array([[-4, 0, +4], [-16, 0, +16], [-4, 0, +4]]) / 432,
        np.array([[-1, 0, +1], [-4, 0, +4], [-1, 0, +1]]) / 432,
    ],
]

DERIVATIVE_4D_X_STENCILS_LIST = [
    STANDARD_4D_X_DERIVATIVE_STENCIL,
    BRILLOUIN_4D_X_DERIVATIVE_STENCIL,
    ISOTROPIC_4D_X_DERIVATIVE_STENCIL,
]
