import sys
import unittest
import itertools as it

import pytest
from hypothesis import example, given, assume, strategies as st
import numpy as np
from numpy.linalg import inv, matrix_power, det

sys.path.append("../..")
import custom_library.constants as constants
import custom_library.auxiliary as auxiliary


@pytest.mark.property_based_test
class TestGammaMatrices(unittest.TestCase):
    def test_definition_of_gamma5_matrix(self):
        """Tests product γ5=γ0γ1γ2γ3."""

        gamma_matrices_product = np.identity(4)
        for index in range(4):
            gamma_matrices_product = np.matmul(
                gamma_matrices_product, constants.GAMMA_MATRICES[index]
            )

        tested_expression = gamma_matrices_product

        expected_expression = constants.GAMMA_MATRICES[4]

        assert np.all(tested_expression == expected_expression)

    @given(gamma_matrix_index=st.integers(min_value=1, max_value=constants.SU2_DOF))
    def test_Hermiticity_of_gamma_matrices(self, gamma_matrix_index):
        """Tests γ_i^†=γ_i."""

        tested_expression = np.conjugate(
            np.transpose(constants.GAMMA_MATRICES[gamma_matrix_index])
        )

        expected_expression = constants.GAMMA_MATRICES[gamma_matrix_index]

        assert np.all(tested_expression == expected_expression)

    @given(gamma_matrix_index=st.integers(min_value=1, max_value=constants.SU2_DOF))
    def test_inverse_of_gamma_matrices(self, gamma_matrix_index):
        """Tests γ_i^-1=γ_i."""

        tested_expression = inv(constants.GAMMA_MATRICES[gamma_matrix_index])

        expected_expression = constants.GAMMA_MATRICES[gamma_matrix_index]

        assert np.all(tested_expression == expected_expression)

    @given(
        gamma_matrix_index_mu=st.integers(min_value=1, max_value=constants.SU2_DOF),
        gamma_matrix_index_nu=st.integers(min_value=1, max_value=constants.SU2_DOF),
    )
    def test_gamma_matrices_anticommutation_relation(
        self, gamma_matrix_index_mu, gamma_matrix_index_nu
    ):
        """Tests anticommutation relation for Euclidean gamma matrices: {g_mu, g_nu} = 2 d_mu,nu I_4, with: mu, nu = 0, ..., 4."""

        tested_expression = auxiliary.matrix_anticommutation_relation(
            constants.GAMMA_MATRICES[gamma_matrix_index_mu],
            constants.GAMMA_MATRICES[gamma_matrix_index_nu],
        )

        expected_expression = (
            2
            * auxiliary.Kronecker_delta(gamma_matrix_index_nu, gamma_matrix_index_mu)
            * np.identity(4)
        )

        assert np.all(tested_expression == expected_expression)


@pytest.mark.property_based_test
class TestPauliMatrices(unittest.TestCase):
    @given(Paul_matrix_index=st.integers(min_value=1, max_value=constants.SU2_DOF))
    def test_square_of_Pauli_matrices(self, Paul_matrix_index):
        """Tests σ_1^2 = σ_2^2 = σ_3^2 = I_2."""

        tested_expression = matrix_power(constants.PAULI_MATRICES[Paul_matrix_index], 2)

        expected_expression = np.identity(2, dtype=np.complex_)

        assert np.all(tested_expression == expected_expression)

    @given(Paul_matrix_index=st.integers(min_value=1, max_value=constants.SU2_DOF))
    def test_trace_of_Pauli_matrices(self, Paul_matrix_index):
        """Tests Tr[σ_i] = 0."""

        tested_expression = np.trace(constants.PAULI_MATRICES[Paul_matrix_index])

        expected_expression = 0

        assert tested_expression == expected_expression

    @given(Paul_matrix_index=st.integers(min_value=1, max_value=constants.SU2_DOF))
    def test_determinant_of_Pauli_matrices(self, Paul_matrix_index):
        """Tests Det[σ_i] = -1."""

        tested_expression = det(constants.PAULI_MATRICES[Paul_matrix_index])

        expected_expression = -1

        assert tested_expression == expected_expression

    def test_product_of_Pauli_matrices(self):
        """Tests product -i*s1 σ_2 σ_3 = I_2."""

        Pauli_matrices_product = np.identity(2)
        for index in [1, 2, 3]:
            Pauli_matrices_product = np.matmul(
                Pauli_matrices_product, constants.PAULI_MATRICES[index]
            )

        tested_expression = -1.0j * Pauli_matrices_product

        expected_expression = constants.PAULI_MATRICES[0]

        assert np.all(tested_expression == expected_expression)

    @given(
        Paul_matrix_index_i=st.integers(min_value=1, max_value=constants.SU2_DOF),
        Paul_matrix_index_j=st.integers(min_value=1, max_value=constants.SU2_DOF),
    )
    def test_commutation_relation_of_Pauli_matrices(
        self, Paul_matrix_index_i, Paul_matrix_index_j
    ):
        """Tests commutation relation for Euclidean gamma matrices: [σ_i, σ_j] = 2 i epsilon_ijk σ_k, with: i, j, k = 0, ..., 2."""

        tested_expression = auxiliary.matrix_commutation_relation(
            constants.PAULI_MATRICES[Paul_matrix_index_i],
            constants.PAULI_MATRICES[Paul_matrix_index_j],
        )

        expected_expression = 2.0j * sum(
            [
                auxiliary.Levi_Civita_symbol(
                    Paul_matrix_index_i, Paul_matrix_index_j, Paul_matrix_index_k
                )
                * constants.PAULI_MATRICES[Paul_matrix_index_k]
                for Paul_matrix_index_k in range(4)
            ]
        )

        assert np.all(tested_expression == expected_expression)

    @given(
        Paul_matrix_index_i=st.integers(min_value=1, max_value=constants.SU2_DOF),
        Paul_matrix_index_j=st.integers(min_value=1, max_value=constants.SU2_DOF),
    )
    def test_anticommutation_relation_of_Pauli_matrices(
        self, Paul_matrix_index_i, Paul_matrix_index_j
    ):
        """Tests anticommutation relation for Pauli matrices: {σ_i, σ_j} = 2 d_i,j I_2, with: i, j = 0, ..., 2."""

        tested_expression = auxiliary.matrix_anticommutation_relation(
            constants.PAULI_MATRICES[Paul_matrix_index_i],
            constants.PAULI_MATRICES[Paul_matrix_index_j],
        )

        expected_expression = (
            2
            * auxiliary.Kronecker_delta(Paul_matrix_index_i, Paul_matrix_index_j)
            * constants.PAULI_MATRICES[0]
        )

        assert np.all(tested_expression == expected_expression)


@pytest.mark.property_based_test
class TestLambdaMatrices(unittest.TestCase):
    @given(lambda_matrix_index=st.integers(min_value=1, max_value=constants.SU3_DOF))
    def test_Hermiticity_of_Lambda_matrices(self, lambda_matrix_index):
        """Tests (l_a)^† = l_a."""

        tested_expression = np.transpose(
            np.conjugate(constants.LAMBDA_MATRICES[lambda_matrix_index])
        )

        expected_expression = constants.LAMBDA_MATRICES[lambda_matrix_index]

        assert np.all(tested_expression == expected_expression)

    @given(lambda_matrix_index=st.integers(min_value=1, max_value=constants.SU3_DOF))
    def test_trace_of_Lambda_matrices(self, lambda_matrix_index):
        """Tests Tr[l_a] = 0."""

        tested_expression = np.trace(constants.LAMBDA_MATRICES[lambda_matrix_index])

        expected_expression = 0

        assert tested_expression == expected_expression

    @given(
        lambda_matrix_index_a=st.integers(min_value=1, max_value=constants.SU3_DOF),
        lambda_matrix_index_b=st.integers(min_value=1, max_value=constants.SU3_DOF),
    )
    def test_trace_of_product_of_two_Lambda_matrices(
        self, lambda_matrix_index_a, lambda_matrix_index_b
    ):
        """Tests Tr[l_a l_b] = 2 d_ab."""

        tested_expression = np.trace(
            np.matmul(
                constants.LAMBDA_MATRICES[lambda_matrix_index_a],
                constants.LAMBDA_MATRICES[lambda_matrix_index_b],
            )
        )

        expected_expression = 2 * auxiliary.Kronecker_delta(
            lambda_matrix_index_a, lambda_matrix_index_b
        )

        assert np.isclose(tested_expression, expected_expression)

    @given(
        index=st.integers(min_value=0, max_value=511)
    )  # 512 is total number of combinations of 3 indices in the range [1,8]
    def test_reproduction_of_structure_constants(self, index):
        """Tests f_abc = (-1/4 i)*Tr[l_a [l_b, l_c]."""

        # Generating a list with a all combinations of indices
        indices_tuple = list(
            set(it.product(range(1, constants.SU3_DOF + 1), repeat=3))
        )[index]

        (
            lambda_matrix_index_a,
            lambda_matrix_index_b,
            lambda_matrix_index_c,
        ) = indices_tuple

        tested_expression = (-1.0j / 4) * np.trace(
            np.matmul(
                constants.LAMBDA_MATRICES[lambda_matrix_index_a],
                auxiliary.matrix_commutation_relation(
                    constants.LAMBDA_MATRICES[lambda_matrix_index_b],
                    constants.LAMBDA_MATRICES[lambda_matrix_index_c],
                ),
            )
        )

        expected_expression = constants.SU3_STRUCTURE_CONSTANTS_DICTIONARY.get(
            indices_tuple, 0
        )

        assert np.isclose(tested_expression, expected_expression)

    # @pytest.mark.skip
    @given(
        lambda_matrix_index_a=st.integers(min_value=0, max_value=constants.SU3_DOF),
        lambda_matrix_index_b=st.integers(min_value=0, max_value=constants.SU3_DOF),
    )
    def test_commutation_relation_of_Lambda_matrices(
        self, lambda_matrix_index_a, lambda_matrix_index_b
    ):
        """Tests [l_a, l_b] = 2i*f_abc l_c."""

        tested_expression = auxiliary.matrix_commutation_relation(
            constants.LAMBDA_MATRICES[lambda_matrix_index_a],
            constants.LAMBDA_MATRICES[lambda_matrix_index_b],
        )

        expected_expression = 2.0j * sum(
            [
                constants.SU3_STRUCTURE_CONSTANTS_DICTIONARY.get(
                    (
                        lambda_matrix_index_a,
                        lambda_matrix_index_b,
                        lambda_matrix_index_c,
                    ),
                    0,
                )
                * constants.LAMBDA_MATRICES[lambda_matrix_index_c]
                for lambda_matrix_index_c in range(1, constants.SU3_DOF + 1)
            ]
        )

        assert np.allclose(tested_expression, expected_expression)

    @given(
        lambda_matrix_index_a=st.integers(min_value=1, max_value=constants.SU3_DOF),
        lambda_matrix_index_b=st.integers(min_value=1, max_value=constants.SU3_DOF),
    )
    def test_anticommutation_relation_of_Lambda_matrices(
        self, lambda_matrix_index_a, lambda_matrix_index_b
    ):
        """Tests {l_a, l_b} = 4/3 d_ab I + 2 d_abc l_c."""

        # Helper function
        def d_abc(a, b, c):
            return (1 / 4) * np.trace(
                np.matmul(
                    constants.LAMBDA_MATRICES[a],
                    auxiliary.matrix_anticommutation_relation(
                        constants.LAMBDA_MATRICES[b], constants.LAMBDA_MATRICES[c]
                    ),
                )
            )

        tested_expression = auxiliary.matrix_anticommutation_relation(
            constants.LAMBDA_MATRICES[lambda_matrix_index_a],
            constants.LAMBDA_MATRICES[lambda_matrix_index_b],
        )

        expected_expression = (4 / 3) * auxiliary.Kronecker_delta(
            lambda_matrix_index_a, lambda_matrix_index_b
        ) * constants.LAMBDA_MATRICES[0] + 2 * sum(
            [
                d_abc(
                    lambda_matrix_index_a, lambda_matrix_index_b, lambda_matrix_index_c
                )
                * constants.LAMBDA_MATRICES[lambda_matrix_index_c]
                for lambda_matrix_index_c in range(1, constants.SU3_DOF + 1)
            ]
        )

        assert np.allclose(tested_expression, expected_expression)

    def test_sum_of_squares_of_Lambda_matrices(self):
        """Tests Casimir operator: l_i l_i = (16/3) I_3."""

        tested_expression = sum(
            [
                matrix_power(constants.LAMBDA_MATRICES, 2)[lambda_matrix_index]
                for lambda_matrix_index in range(1, constants.SU3_DOF + 1)
            ]
        )

        expected_expression = (16 / 3) * np.identity(3)

        assert np.allclose(tested_expression, expected_expression)


@pytest.mark.property_based_test
class TestStencils(unittest.TestCase):
    # LAPLACIAN STENCILS

    @given(index=st.integers(min_value=0, max_value=3))
    def test_2D_Laplacian_stencils(self, index):
        """Tests symmetry of the 2D Laplacian stencil."""

        Laplacian_stencil = constants.LAPLACIAN_2D_STENCILS_LIST[index]

        assert np.allclose(Laplacian_stencil, np.transpose(Laplacian_stencil))

    @given(index=st.integers(min_value=0, max_value=3))
    def test_3D_Laplacian_stencils(self, index):
        """Tests symmetry of the 3D Laplacian stencil."""

        Laplacian_stencil = constants.LAPLACIAN_3D_STENCILS_LIST[index]

        assert np.allclose(Laplacian_stencil, np.transpose(Laplacian_stencil))

    @given(index=st.integers(min_value=0, max_value=3))
    def test_4D_Laplacian_stencils(self, index):
        """Tests symmetry of the 4D Laplacian stencil."""

        Laplacian_stencil = constants.LAPLACIAN_4D_STENCILS_LIST[index]

        assert np.allclose(Laplacian_stencil, np.transpose(Laplacian_stencil))

    # DERIVATIVE STENCILS

    @given(index=st.integers(min_value=0, max_value=2))
    def test_2D_derivative_stencils(self, index):
        """Tests anti-symmetry of the 2D Laplacian stencil."""

        derivative_stencil = constants.DERIVATIVE_2D_X_STENCILS_LIST[index]

        assert np.allclose(derivative_stencil, -np.flip(derivative_stencil, axis=1))

    @given(index=st.integers(min_value=0, max_value=2))
    def test_3D_derivative_stencils(self, index):
        """Tests anti-symmetry of the 3D Laplacian stencil."""

        derivative_stencil = constants.DERIVATIVE_3D_X_STENCILS_LIST[index]

        assert np.allclose(derivative_stencil, np.flip(derivative_stencil, axis=0))

    @given(index=st.integers(min_value=0, max_value=2))
    def test_4D_derivative_stencils(self, index):
        """Tests anti-symmetry of the 4D Laplacian stencil."""

        derivative_stencil = constants.DERIVATIVE_4D_X_STENCILS_LIST[index]

        assert np.allclose(derivative_stencil, np.flip(derivative_stencil, axis=0))

    @given(index=st.integers(min_value=0, max_value=2))
    def test_4D_derivative_stencils(self, index):
        """Tests anti-symmetry of the 4D Laplacian stencil."""

        derivative_stencil = constants.DERIVATIVE_4D_X_STENCILS_LIST[index]

        assert np.allclose(derivative_stencil, np.flip(derivative_stencil, axis=1))


if __name__ == "__main__":
    unittest.main()
