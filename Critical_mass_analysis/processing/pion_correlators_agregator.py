'''
TODO: Write a more detailed description.
...
'''
import numpy as np
import click
import os.path
import sys
import re

# custom modules
sys.path.append('../')
import custom_library.auxiliary as auxiliary


@click.command()
@click.option("--input_directory", "input_directory", "-in_dir", default='./to_be_processed/', help="The directory of the files to be concatenated.")
@click.option("--output_directory", "output_directory", "-out_dir", default='../processed_files/', help="The directory that will contain the output concatenated binary files with the time-dependent pion correlator arrays")

def main(input_directory, output_directory):

    assert os.path.isdir(input_directory) and (len(os.listdir(input_directory)) > 0), 'Input directory invalid or empty'
    auxiliary.creating_directory_function(output_directory)

    pion_correlators_dictionary = dict()
    for filename in os.listdir(os.fsencode(input_directory)):
        filename = os.fsdecode(filename)

        # Extract info from each file
        operator_type_label = re.findall(r'^.+_operator', filename)[0]
        laplacian_stencil_label = re.findall(r'operator_(.+_laplacian)', filename)[0]
        derivative_stencil_label = re.findall(r'laplacian_(.+_derivative)', filename)[0]
        lattice_size = int((re.findall(r'_L=(\d+)', filename))[0])
        bare_mass_value = float(re.findall(r'mb=(\d*\.\d+)', filename)[0])
        CG_precision = re.findall(r'CG=(\d*e-\d+)', filename)[0]
        number_of_SF_iterations = (re.findall(r'_NSF=(\d+)', filename))
        precision_info = 'CG='+CG_precision
        if (len(number_of_SF_iterations) > 0):
            number_of_SF_iterations = int(number_of_SF_iterations[0])
            precision_info += f'_NSF={number_of_SF_iterations}'
        number_of_KL_iterations = (re.findall(r'_KL_iter=(\d+)', filename))
        if (len(number_of_KL_iterations) > 0):
            number_of_KL_iterations = int(number_of_KL_iterations[0])
            precision_info += f'_NSF={number_of_KL_iterations}'
        
        # Pass to dictionary for distinguishing between operator types and specification
        operator_specification = laplacian_stencil_label+'_'+derivative_stencil_label
        pion_correlators_specification_key = operator_type_label, operator_specification, 'L=' + str(lattice_size), precision_info, bare_mass_value
        if (pion_correlators_dictionary.get(pion_correlators_specification_key) == None):
            # initialize pion_correlators_specification_key if value is None
            pion_correlators_dictionary[pion_correlators_specification_key] = list()
        pion_correlators_dictionary[pion_correlators_specification_key].append(filename)

    for pion_correlators_specification_key in pion_correlators_dictionary.keys():

        # Test for any discrepancies in the number of processed configurations
        total_number_of_processed_configurations = 0
        for filename in pion_correlators_dictionary[pion_correlators_specification_key]:
            number_of_processed_configurations = int((re.findall(r'_configs_total=(\d+)', filename))[0])
            initial_configuration_index = int((re.findall(r'_initial_config=(\d+)', filename))[0])
            total_number_of_processed_configurations += number_of_processed_configurations
        assert total_number_of_processed_configurations <= 100, 'Total number of processed configurations must be smaller than 100.'

        # Construct the aggregated output array
        pion_correlators_array = list()
        for filename in pion_correlators_dictionary[pion_correlators_specification_key]:
            pion_correlators_array.append(np.fromfile(input_directory+filename, dtype=np.float_))
        output_array = np.concatenate(pion_correlators_array)

        # configuring subdirectories structure
        output_subdirectories = output_directory
        for subdirectory in pion_correlators_specification_key[:-1]:
            output_subdirectories += subdirectory + '/'
            auxiliary.creating_directory_function(output_subdirectories)

        # Pass output array to binary file
        bare_mass_value = pion_correlators_specification_key[-1]
        output_filename = f'mb={bare_mass_value:.2f}'+f'_configs_total={total_number_of_processed_configurations}'
        output_array.tofile(output_subdirectories+output_filename)

        print('Binary file:', output_filename, 'was created inside directory', output_subdirectories)

if __name__ == "__main__":
    main()
