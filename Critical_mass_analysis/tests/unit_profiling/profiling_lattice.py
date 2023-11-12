import sys

sys.path.append('../../')
import custom_library.auxiliary as auxiliary
import custom_library.lattice as lattice


@auxiliary.PerformanceTracker(skip_boolean=False) 
def profiling_lattice_coordinates_vectors_addition():
    '''
    NOTE: Current timing of 'lattice_coordinates_vectors_addition' average run time (for 10 repetitions): 4.53±2.43 μs.
    '''

    # Timer specifications
    REPETITIONS=1000
    TIME_UNITS_LABEL='μs'
    PRINT_RESULT_BOOLEAN = True
    timer_specifications = [REPETITIONS, TIME_UNITS_LABEL, PRINT_RESULT_BOOLEAN]

    execution_timer = auxiliary.ExecutionTimer(*timer_specifications)
    lattice_structure_instance = lattice.LatticeStructure()

    tuple_a = (8, 7)
    tuple_b = (7, 6)
    
    input_parameters = tuple_a, tuple_b

    @auxiliary.ExecutionTimer(*timer_specifications)
    def alternative_method_1(input_parameters):
        a, b = input_parameters

        return tuple(map(lambda components: sum(components)%components[2], zip(a, b, lattice_structure_instance.lattice_shape)))

    alternative_method_1(input_parameters)

    @auxiliary.ExecutionTimer(*timer_specifications)
    def alternative_method_2(input_parameters):
        a, b = input_parameters

        lattice_shape = lattice_structure_instance.lattice_shape

        return tuple([ (a[index] + b[index] + lattice_shape[index])%lattice_shape[index] for index in range(lattice_structure_instance.lattice_dimensions) ])
    
    alternative_method_2(input_parameters)

    @auxiliary.ExecutionTimer(*timer_specifications)
    def alternative_method_3(input_parameters):
        a, b = input_parameters

        lattice_shape = lattice_structure_instance.lattice_shape
        
        lattice_sites_coordinates_sum = tuple()
        for index in range(lattice_structure_instance.lattice_dimensions):
            lattice_sites_coordinates_sum += ( ((tuple_a[index]+tuple_b[index]+lattice_shape[index])%lattice_shape[index]), )
        
        return lattice_sites_coordinates_sum
    
    alternative_method_3(input_parameters)

    @auxiliary.ExecutionTimer(*timer_specifications)
    def alternative_method_4(input_parameters):
        a, b = input_parameters

        lattice_shape = lattice_structure_instance.lattice_shape

        return tuple((a + b + L)%L for a, b, L in zip(a, b, lattice_shape))
    
    alternative_method_4(input_parameters)

    
    tested_method = execution_timer(lattice_structure_instance.lattice_coordinates_vectors_addition)
    tested_method(*input_parameters)


if __name__ == "__main__":
    main_performance_tracker = auxiliary.PerformanceTracker()
