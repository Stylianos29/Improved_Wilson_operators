'''
TODO: Write a more detailed description.
...
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import gvar as gv
import lsqfit
import os
import re
import sys
import click

# custom modules
sys.path.append('../')
import custom_library.auxiliary as auxiliary
import custom_library.post_processing as post_processing


@click.command()
@click.option("--input_directory", "input_directory", "-in_dir", default='./effective_mass_estimates/', help="The directory of the files to be analyzed.")
@click.option("--output_directory", "output_directory", "-out_dir", default='./effective_mass_estimates/', help="The directory storing files with the analyzed effective mass estimates.")
@click.option("--plotting_directory", "plotting_directory", "-plot_dir", default='../plots/', help="The directory that will contain the analyzed plots.")

def main(input_directory, output_directory, plotting_directory):

    # Extract the subdirectories structure of the input directory
    complete_list_of_subdirectories = [subdirectory[0] for subdirectory in os.walk(input_directory)]
    processed_files_subdirectories = [subdirectory for subdirectory in complete_list_of_subdirectories if 'CG' in subdirectory]

    critical_mass_dictionary = dict()
    for processed_files_subdirectory in processed_files_subdirectories:

        print(processed_files_subdirectory)

        # Extract pieces of info from each subdirectory
        operator_type_label = re.findall(r'^(.+_operator)', processed_files_subdirectory.replace(input_directory,''))[0]
        laplacian_stencil_label = re.findall(r'operator/(.+_laplacian)', processed_files_subdirectory)[0]
        derivative_stencil_label = re.findall(r'laplacian_(.+_derivative)', processed_files_subdirectory)[0]
        operator_specification_label = laplacian_stencil_label+'_'+derivative_stencil_label
        lattice_size = int((re.findall(r'L=(\d+)', processed_files_subdirectory))[0])
        # precision
        CG_precision = re.findall(r'CG=(\d*e-\d+)', processed_files_subdirectory)[0]
        number_of_SF_iterations = (re.findall(r'NSF=(\d+)', processed_files_subdirectory))
        number_of_KL_iterations = (re.findall(r'KL_iter=(\d+)', processed_files_subdirectory))
        precision_info = 'CG='+CG_precision
        if (len(number_of_SF_iterations) > 0):
            precision_info += f'_NSF={number_of_SF_iterations[0]}'
        if (len(number_of_KL_iterations) > 0):
            precision_info += f'_KL_iter={number_of_KL_iterations[0]}'

        jackknife_effective_mass_estimates_dictionary = dict() # per configuration per bare mass value
        for filename in os.listdir(os.fsencode(processed_files_subdirectory)):
            filename = os.fsdecode(filename)
            
            # Extract pieces of info from each file
            bare_mass_value = float(re.findall(r'mb=(\d*\.\d+)', filename)[0])

            # Import array effective mass estimates array per bare mass values
            jackknife_effective_mass_estimates_dictionary[bare_mass_value] = np.fromfile(processed_files_subdirectory+'/'+filename, dtype=np.float64)

        if (len(jackknife_effective_mass_estimates_dictionary.keys()) < 3):
            continue

        bare_mass_value_array = np.array(list(jackknife_effective_mass_estimates_dictionary.keys()))
        common_statistical_error_array = gv.sqrt(list(map(post_processing.jackknife_variance_array_function, np.array(list(jackknife_effective_mass_estimates_dictionary.values())))))
        temp_array = np.transpose(np.array(list(jackknife_effective_mass_estimates_dictionary.values())))

        number_of_processed_configurations = len(temp_array)

        x = bare_mass_value_array

        linear_fit_parameters_per_configuration_list = list()
        for index in range(len(temp_array)):

            y = gv.gvar(temp_array[index], common_statistical_error_array)
            y = np.square(y)

            # linear fit parameter guess
            slope_guess = (np.max(gv.mean(y))-np.min(gv.mean(y)))/(np.max(x)-np.min(x))
            linear_p0 = [ slope_guess, -np.min(gv.mean(y))/slope_guess ]
            # linear fit per configuration
            linear_fit = lsqfit.nonlinear_fit(data=(bare_mass_value_array, y), p0=linear_p0, fcn=post_processing.linear_func, debug=True)
            linear_fit_parameters = linear_fit.p

            linear_fit_parameters_per_configuration_list.append(gv.mean(linear_fit_parameters))

        jackknife_averages_of_linear_fit_parameters = gv.gvar(zip(np.average(linear_fit_parameters_per_configuration_list, axis=0), np.sqrt(list(map(post_processing.jackknife_variance_array_function, np.transpose(linear_fit_parameters_per_configuration_list))))))

        critical_mass_dictionary[operator_type_label, operator_specification_label, lattice_size, precision_info] = jackknife_averages_of_linear_fit_parameters[1] # per dataset

        squared_jackknife_averages_of_effective_mass_estimates_array = np.square(gv.gvar(list(zip(np.average(temp_array, axis=0), common_statistical_error_array))))

        # Plotting the optimum effective mass estimates against the bare mass values
        fig, ax = plt.subplots()
        ax.grid()
        ax.set_title('Squared effective mass for various bare mass values (L='+str(lattice_size)+', N='+str(number_of_processed_configurations)+')'+'\n['+operator_type_label.replace('_',' ')+' '+operator_specification_label.replace('_',' ')+']', pad = 6)
        ax.set(xlabel='$a m_b$', ylabel='$m^2_{eff.}$')
        # Axes lines
        ax.axhline(0, color='black') # x = 0
        ax.axvline(0, color='black') # y = 0
        plt.errorbar(bare_mass_value_array, gv.mean(squared_jackknife_averages_of_effective_mass_estimates_array), yerr=gv.sdev(squared_jackknife_averages_of_effective_mass_estimates_array), fmt='o', markersize=8, capsize=10)
        # Linear best-fit line
        critical_mass_value = jackknife_averages_of_linear_fit_parameters[1]
        x_data = np.linspace(critical_mass_value.mean-0.005, np.max(bare_mass_value_array)+0.005, 50)
        y_data = post_processing.linear_func(x_data, gv.mean(jackknife_averages_of_linear_fit_parameters))
        ax.plot(x_data, gv.mean(y_data), 'r-', label='$a m_c$ = {:.4f}'.format(critical_mass_value.mean)+u"\u00B1"+'{:.4f}'.format(critical_mass_value.sdev)+'\n$\kappa_c$ = {:.4f}'.format(post_processing.critical_kappa_value_function(critical_mass_value).mean)+u"\u00B1"+'{:.4f}'.format(post_processing.critical_kappa_value_function(critical_mass_value).sdev) +f'\n $χ^2$/dof={linear_fit.chi2:.4}/{linear_fit.dof}={linear_fit.chi2/linear_fit.dof:.4}' )
        ax.fill_between(x_data, gv.mean(y_data) - gv.sdev(y_data), gv.mean(y_data) + gv.sdev(y_data), color='r', alpha=0.2)

        # quadratic fit
        if (len(jackknife_effective_mass_estimates_dictionary.keys()) >= 4):
            # quadratic fit parameter guess
            quadratic_p0 = [0.01, *gv.mean(jackknife_averages_of_linear_fit_parameters)]
            quadratic_fit = lsqfit.nonlinear_fit(data=(bare_mass_value_array, squared_jackknife_averages_of_effective_mass_estimates_array), p0=quadratic_p0, fcn=post_processing.quadratic_func, debug=True)
            quadratic_fit_parameters = quadratic_fit.p
            # plotting quadratic best-fit line
            y_data = post_processing.quadratic_func(x_data, quadratic_fit_parameters)
            ax.plot(x_data, gv.mean(y_data), 'g-')
            ax.fill_between(x_data, gv.mean(y_data) - gv.sdev(y_data), gv.mean(y_data) + gv.sdev(y_data), color='g', alpha=0.2)

        ax.legend(loc="upper left")
        fig.savefig(processed_files_subdirectory.replace('./effective_mass_estimates/', plotting_directory)+'/Effective_mass_values_for_various_bare_mass_values.png')

        plt.close()

    critical_mass_values_parameters = np.transpose(list(critical_mass_dictionary.keys()))

    operator_type_label_set = list(set(critical_mass_values_parameters[0]))
    operator_specification_label_set = list(set(critical_mass_values_parameters[1]))
    lattice_size_set = list(set(critical_mass_values_parameters[2]))
    precision_info_set = list(set(critical_mass_values_parameters[3]))

    temp_dictionary = dict()
    for dataset_specification in critical_mass_dictionary.keys():

        print(dataset_specification)

        operator_type_label, operator_specification_label, lattice_size, precision_info = dataset_specification

        if ((operator_type_label == 'Improved_Wilson_operator') and (operator_specification_label == 'standard_laplacian_standard_derivative') and (precision_info == 'CG=1e-06')):

            temp_dictionary[int(lattice_size)] = critical_mass_dictionary[dataset_specification]

    print(list(temp_dictionary.keys()))

    # Plotting the optimum effective mass estimates against the bare mass values
    fig, ax = plt.subplots()
    ax.grid()
    plt.errorbar(list(temp_dictionary.keys()), gv.mean(list(temp_dictionary.values())), yerr=gv.sdev(list(temp_dictionary.values())), fmt='o', markersize=8, capsize=10)
    fig.savefig('./test.png')
    plt.close()

print("Done!")


if __name__ == "__main__":
    main()
