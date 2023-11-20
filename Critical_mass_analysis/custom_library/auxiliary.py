import itertools as it
import os
import inspect
import functools
import time
from timeit import timeit
from cProfile import Profile

import numpy as np
import dis
from pstats import SortKey, Stats
from hypothesis import example, given, assume, strategies as st

import custom_library.constants as constants
import custom_library.lattice as lattice


# MATHEMATICAL FUNCTIONS


def Kronecker_delta(i, j):
    return 1 if i == j else 0


def Levi_Civita_symbol(i, j, k):
    # Define the permutation of (1, 2, 3) and its sign
    permutation = [(1, 2, 3), (2, 3, 1), (3, 1, 2), (1, 3, 2), (3, 2, 1), (2, 1, 3)]
    sign = [1, 1, 1, -1, -1, -1]

    # Check if the input indices form a valid permutation
    if (i, j, k) in permutation:
        return sign[permutation.index((i, j, k))]
    else:
        return 0


def matrix_commutation_relation(matrix_A, matrix_B):
    assert (
        isinstance(matrix_A, np.ndarray)
        and isinstance(matrix_B, np.ndarray)
        and (matrix_A.shape == matrix_B.shape)
    ), "Input matrices must NumPy array of the same shape."

    return np.matmul(matrix_A, matrix_B) - np.matmul(matrix_B, matrix_A)


def matrix_anticommutation_relation(matrix_A, matrix_B):
    assert (
        isinstance(matrix_A, np.ndarray)
        and isinstance(matrix_B, np.ndarray)
        and (matrix_A.shape == matrix_B.shape)
    ), "Input matrices must NumPy array of the same shape."

    return np.matmul(matrix_A, matrix_B) + np.matmul(matrix_B, matrix_A)


def structure_constants(a, b, c):
    """Return the structure constant value corresponding the input tuple of indices."""

    input_indices_tuple = (a, b, c)
    if input_indices_tuple in constants.SU3_STRUCTURE_CONSTANTS_DICTIONARY:
        return constants.SU3_STRUCTURE_CONSTANTS_DICTIONARY[input_indices_tuple]

    return 0.0


# It needs to be removed
COMMON_TYPE_STRATEGIES = (
    st.text()
    | st.none()
    | st.integers()
    | st.dates()
    | st.floats()
    | st.complex_numbers()
    | st.tuples()
    | st.booleans()
)


@st.composite
def generate_lattice_shapes(draw, max_value=15):
    """Generates a random tuple to be interpreted as a lattice shape of the form: (lattice_size >= 9, lattice_dimensions in [1,2,3,4], temporal_axis_size >= 9 or None), numerical values being all integers."""

    lattice_size = draw(st.integers(min_value=9, max_value=15))
    lattice_dimensions = draw(st.integers(min_value=1, max_value=4))
    temporal_axis_size = draw(st.integers(min_value=9, max_value=15) | st.none())

    lattice_shape = (
        lattice.LatticeStructure.turning_fundamental_parameters_to_lattice_shape(
            lattice_size=lattice_size,
            lattice_dimensions=lattice_dimensions,
            temporal_axis_size=temporal_axis_size,
        )
    )

    return lattice_shape


@st.composite
def tuples_of_numbers(
    draw, numbers_type="int", min_size=0, max_size=5, min_value=0, max_value=5
):
    """Generates variable-sized tuples of numbers within an adjustable range."""

    if numbers_type == "int":
        elements = st.integers(min_value=min_value, max_value=max_value)

    elif numbers_type == "float":
        elements = st.floats(min_value=min_value, max_value=max_value)

    else:
        AssertionError(
            'Input value of the "numbers_type" argument must be either a "int" or a "float" string.'
        )

    lists_of_various_size = draw(
        st.lists(elements, min_size=min_size, max_size=max_size)
    )

    return tuple(lists_of_various_size)


def generate_common_strategies_excluding(list_of_excluded_types=list()):
    """Generates union of common type strategies."""

    # Simpler strategies have been put first for better shrinking
    COMMON_TYPE_STRATEGIES_DICTIONARY = {
        "none": st.none(),
        "integers": st.integers(),
        "booleans": st.booleans(),
        "floats": st.floats(),
        "complex numbers": st.complex_numbers(),
        "dates": st.dates(),
        "text": st.text(),
        "tuples of integers": tuples_of_numbers(numbers_type="int"),
        "tuples of reals": tuples_of_numbers(numbers_type="float"),
    }

    if not isinstance(list_of_excluded_types, list):
        list_of_excluded_types = [list_of_excluded_types]
    assert all(
        excluded_type in COMMON_TYPE_STRATEGIES_DICTIONARY
        for excluded_type in list_of_excluded_types
    ), "The list of excluded types passed contains invalid elements."

    list_of_included_types = list()
    for strategy_type, strategy in COMMON_TYPE_STRATEGIES_DICTIONARY.items():
        if strategy_type not in list_of_excluded_types:
            list_of_included_types.append(strategy)

    composite_strategy = st.one_of(list_of_included_types)

    return composite_strategy


########################################


def matrix_extended_gamma5_function(matrix=None, matrix_side_length=None):
    if matrix is not None:
        matrix = np.array(matrix)
        matrix_side_length = matrix.shape[0]
    elif matrix_side_length is not None:
        matrix_side_length = matrix_side_length

    assert matrix_side_length % 4 == 0, "Passed matrix is not a Dirac operator."

    return np.kron(np.identity(matrix_side_length // 4), constants.GAMMA_MATRICES[4])


def relative_approximation_error(approximate_value, exact_value):
    """Calculation of the percent relative error of an approximate quantity."""

    assert isinstance(approximate_value, float) and isinstance(
        exact_value, float
    ), "Input values must be real numbers."

    return 100 * abs((approximate_value - exact_value) / exact_value)


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def creating_nonexistent_directory_function(directory_path):
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)
        print(directory_path + " directory created.")


#####################################
# PROFILING TOOLS
#####################################


def compare_two_matrices(A, B) -> str:
    for precision in range(18):
        if not (np.isclose(A, B, atol=float("1e-" + str(precision))).all()):
            break
    return "Up to precision: 1e-" + str(precision - 1)


class CompareTwoMatrices:
    """"""

    def __init__(self, A, B):
        A = np.array(A)
        B = np.array(B)

        assert (
            A.shape == B.shape
        ), "Input matrices for comparison must have the same shape."
        self.A = A
        self.B = B

        self.difference = A - B

    def maximum_absolute_difference(self):
        for precision in range(18):
            if not (
                np.isclose(self.A, self.B, atol=float("1e-" + str(precision))).all()
            ):
                break
        return "Up to precision: 1e-" + str(precision - 1)

    def cumulative_statistics(self):
        array_of_elements = np.abs((self.difference).reshape(-1))

        self.average_difference = np.average(array_of_elements)
        self.standard_deviation = np.std(array_of_elements, ddof=1)

        return f"{self.average_difference:.2e}±{self.standard_deviation:.2e}"


class PerformanceTracker:
    performance_functions_names_log = list()

    def __init__(self, skip_boolean=False):
        self.skip_boolean = skip_boolean

    def __call__(self, func):
        function_name = f"{func.__name__!r}"

        self.performance_functions_names_log.append(function_name)

        if not self.skip_boolean:
            print("===================================")
            print(function_name.upper() + ":")
            print("===================================\n")
            func()

    def print_list(self):
        print(self.performance_functions_names_log)


class ExecutionTimer:
    """
    TODO: Include functools
    """

    TIME_UNITS_DICTIONARY = {"s": 1, "ms": 1e3, "μs": 1e6, "ns": 1e9}

    alternative_methods_memory_reference_log = list()

    def __init__(
        self,
        repetitions=3,
        time_units_label="ms",
        print_result_boolean=True,
        profiler_boolean=False,
    ):
        assert isinstance(repetitions, int) and (
            repetitions > 0
        ), "Number of repetitions must be a positive integer number."
        self.repetitions = repetitions

        assert (
            time_units_label in self.TIME_UNITS_DICTIONARY
        ), "Time units options are: ms, μs, and ns."
        self.time_units_label = time_units_label
        self.time_units = self.TIME_UNITS_DICTIONARY[time_units_label]

        assert isinstance(print_result_boolean, bool)
        self.print_result_boolean = print_result_boolean

        assert isinstance(profiler_boolean, bool)
        self.profiler_boolean = profiler_boolean

    def __call__(self, func):
        def timer(*args, **kwargs):
            function_name = f"{func.__name__!r}"

            self.alternative_methods_memory_reference_log.append(function_name)

            run_times_list = list()
            for _ in range(self.repetitions):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                end = time.perf_counter()
                run_times_list.append(end - start)

            average_run_time = self.time_units * np.average(run_times_list)
            standard_deviation_of_run_times = self.time_units * np.std(
                run_times_list, ddof=1
            )

            print(
                f"▢ {func.__name__!r} average run time (for {self.repetitions} repetitions): {average_run_time:.2f}±{standard_deviation_of_run_times:.2f} {self.time_units_label}. ▢"
            )
            if self.print_result_boolean:
                print("Method output:", result)
            else:
                print("")

            if self.profiler_boolean:
                with Profile() as profile:
                    print(f"{func() = }")
                    (
                        Stats(profile)
                        .strip_dirs()
                        .sort_stats(SortKey.CUMULATIVE)
                        .print_stats()
                    )  # SortKey.CALLS

            return result

        return timer

    def print_list(self):
        print(self.alternative_methods_memory_reference_log)


class ScanningTracker:
    scanning_functions_names_log = list()

    def __init__(self, skip_boolean=False):
        self.skip_boolean = skip_boolean

    def __call__(self, func):
        function_name = f"{func.__name__!r}"

        self.scanning_functions_names_log.append(function_name)

        if not self.skip_boolean:
            print("===================================")
            print(function_name.upper() + ":")
            print("===================================\n")
            func()

            print("Plot was created.")

    def print_list(self):
        print(self.scanning_functions_names_log)


def get_names(f):
    ins = dis.get_instructions(f)
    for x in ins:
        try:
            if (
                x.opcode == 100
                and "<locals>" in next(ins).argval
                and next(ins).opcode == 132
            ):
                yield next(ins).argrepr
                yield from get_names(x.argval)
        except Exception:
            pass


# Custom exceptions
class ReadOnlyAttributeError(Exception):
    def __init__(self, **kwargs):
        assert (
            len(kwargs) == 1
        ), "ReadOnlyAttributeError exception takes only a single argument."

        # Extract the argument's name and value
        attribute_name, value = kwargs.popitem()

        message = f'attribute "{attribute_name}" contains already a value and it cannot be modified to ({value}).'
        super().__init__(message)


def extract_attribute_name(fermion_dimensions):
    attribute_name = f"{fermion_dimensions=}".split("=")[0]


def list_functions(module):
    return [
        name
        for name in dir(module)
        if callable(getattr(module, name)) and hasattr(module, name) and name[0] != "_"
    ]
