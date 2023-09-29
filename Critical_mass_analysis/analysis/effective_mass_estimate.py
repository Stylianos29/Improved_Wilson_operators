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
@click.option("--input_directory", "input_directory", "-in_dir", default='../processed_files/', help="The directory of the files to be analyzed.")
@click.option("--output_directory", "output_directory", "-out_dir", default='./effective_mass_estimates/', help="The directory storing files with the analyzed effective mass estimates.")
@click.option("--plotting_directory", "plotting_directory", "-plot_dir", default='../plots/', help="The directory that will contain the analyzed plots.")

# Effective_mass_estimates

def main(input_directory, output_directory, plotting_directory):

    # Extract the subdirectories structure of the input directory
    complete_list_of_subdirectories = [subdirectory[0] for subdirectory in os.walk(input_directory)]
    processed_files_subdirectories = [subdirectory for subdirectory in complete_list_of_subdirectories if 'CG' in subdirectory]

    for processed_files_subdirectory in processed_files_subdirectories:
        
        # Recreate the subdirectories structure inside the output and plotting directories
        auxiliary.creating_directory_function(output_directory+processed_files_subdirectory.replace(input_directory, ""))
        auxiliary.creating_directory_function(plotting_directory+processed_files_subdirectory.replace(input_directory, ""))

        # Extract pieces of info from each subdirectory
        operator_type_label = re.findall(r'^.+_operator', processed_files_subdirectory)[0]
        laplacian_stencil_label = re.findall(r'operator/(.+_laplacian)', processed_files_subdirectory)[0]
        derivative_stencil_label = re.findall(r'laplacian_(.+_derivative)', processed_files_subdirectory)[0]

        # Construct plotting subdirectory
        # operator_specification = laplacian_stencil_label.capitalize()+'_'+derivative_stencil_label
        operator_plotting_directory = plotting_directory+processed_files_subdirectory.replace(input_directory, "")
        # plotting_directory+operator_type_label+'/'+operator_specification

        for filename in os.listdir(os.fsencode(processed_files_subdirectory)):
            filename = os.fsdecode(filename)
            
            # Import 2D array with time-dependent Pion correlator data
            time_dependent_pion_correlator_per_configuration_2D_array = np.fromfile(processed_files_subdirectory+'/'+filename, dtype=np.float64)

            # Extract pieces of info from each file
            bare_mass_value = float(re.findall(r'mb=(\d*\.\d+)', filename)[0])
            lattice_size = int((re.findall(r'L=(\d+)', processed_files_subdirectory))[0])
            # Calculate the number_of_processed_configurations
            number_of_processed_configurations = np.shape(time_dependent_pion_correlator_per_configuration_2D_array)[0]//lattice_size

            # Reshape for simplifying calculation
            time_dependent_pion_correlator_per_configuration_2D_array = time_dependent_pion_correlator_per_configuration_2D_array.reshape(number_of_processed_configurations, lattice_size)        

            # Average about its central point because the shape of the curve is expected to be symmetric due to periodic boundary conditions
            time_dependent_pion_correlator_per_configuration_2D_array = 0.5*(time_dependent_pion_correlator_per_configuration_2D_array + np.roll(np.flip(time_dependent_pion_correlator_per_configuration_2D_array, axis=1), axis = 1, shift=+1))
        
            # JACKKNIFE CALCULATIONS
            jackknife_replicas_of_time_dependent_pion_correlator_per_configuration_2D_array = post_processing.jackknife_replicas_generation(time_dependent_pion_correlator_per_configuration_2D_array)

            jackknife_average_time_dependent_pion_correlator_array = np.average(jackknife_replicas_of_time_dependent_pion_correlator_per_configuration_2D_array, axis=0)

            jackknife_variance_effective_mass_per_time_array = post_processing.jackknife_variance_array_function(jackknife_replicas_of_time_dependent_pion_correlator_per_configuration_2D_array)

            jackknife_covariance_matrix_effective_mass_per_time_array = post_processing.jackknife_covariance_matrix_function(jackknife_replicas_of_time_dependent_pion_correlator_per_configuration_2D_array)

            jackknife_time_dependent_pion_correlator_array = gv.gvar(jackknife_average_time_dependent_pion_correlator_array, jackknife_covariance_matrix_effective_mass_per_time_array)

            # Calculate the effective mass values array using specific formula
            '''
            TODO: replace it with map()
            '''
            jackknife_replicas_of_effective_mass_per_time_2D_array = list()
            for row in range(number_of_processed_configurations):
                jackknife_replicas_of_effective_mass_per_time_2D_array.append(post_processing.effective_mass_periodic_case_function(jackknife_replicas_of_time_dependent_pion_correlator_per_configuration_2D_array[row]))
            jackknife_replicas_of_effective_mass_per_time_2D_array = np.array(jackknife_replicas_of_effective_mass_per_time_2D_array)
        
            jackknife_average_effective_mass_per_time_array = np.average(jackknife_replicas_of_effective_mass_per_time_2D_array, axis=0)

            jackknife_covariance_matrix_effective_mass_per_time_array = post_processing.jackknife_covariance_matrix_function(jackknife_replicas_of_effective_mass_per_time_2D_array)

            jackknife_effective_mass_per_time_array = gv.gvar(jackknife_average_effective_mass_per_time_array, jackknife_covariance_matrix_effective_mass_per_time_array)
            
            # Investigating the optimal initial time for plateau curve-fitting. The first value is excluded because it is nonsensical and the last one as well in order to maintain at least 2 point for curve-fitting purposes. In total lattice_size//2-3 initial time values are investigated.
            curve_fitting_estimates = list()
            for initial_time in range(1, len(jackknife_effective_mass_per_time_array)-1):

                # Curve-fitting datasets
                x = range(initial_time, len(jackknife_effective_mass_per_time_array))
                y = jackknife_effective_mass_per_time_array[initial_time:]
                # The initial estimate for the effective mass equals the value 
                p0 = [jackknife_effective_mass_per_time_array[3*len(jackknife_effective_mass_per_time_array)//4].mean]

                fit = lsqfit.nonlinear_fit(data=(x, y), p0=p0, fcn=post_processing.plateau_fit_function, debug=True)

                # Manual calculation of p-value **CHECK AGAIN**
                p_value = 1 - stats.chi2.cdf(fit.chi2, fit.dof)
                curve_fitting_estimates.append([fit.p[0], fit.chi2, fit.dof, p_value])

            curve_fitting_estimates = np.array(curve_fitting_estimates)

            # Creating an array of strings with the chi2/dof values per initial time for plotting purposes
            plateau_fit_chi_square_per_dof_array = list()
            curve_fitting_chi_square_estimates = (curve_fitting_estimates.T)[1]
            curve_fitting_dof_values = (curve_fitting_estimates.T)[2]
            for i in range(len(curve_fitting_chi_square_estimates)):
                plateau_fit_chi_square_per_dof_array.append("{:.3f}".format(curve_fitting_chi_square_estimates[i]/curve_fitting_dof_values[i]))

            # Array with the plateau fit effective mass estimates per initial time
            curve_fitting_effective_mass_estimates_array = (curve_fitting_estimates.T)[0]

            # Estimating the optimal effective mass estimate defined as the value with a corresponding p-value closest to the critical value 5%.
            curve_fitting_p_values_array = (curve_fitting_estimates.T)[3]
            curve_fitting_p_values_array = curve_fitting_p_values_array[:-2]

            if (len(curve_fitting_p_values_array) == 0):
                continue
            
            # METHOD 1
            # shifted_curve_fitting_p_values_array = np.abs(curve_fitting_p_values_array - 0.025)
            # optimum_effective_mass_estimate_index = len(shifted_curve_fitting_p_values_array) - 1 - np.argmin( np.flip(shifted_curve_fitting_p_values_array) )
            
            # METHOD 2
            # print(np.max(curve_fitting_p_values_array))
            optimum_effective_mass_estimate_index = np.argmax(curve_fitting_p_values_array)
            # print(np.argmax(curve_fitting_p_values_array))
            # print(optimum_effective_mass_estimate_index)
            # print()

            optimum_effective_mass_estimate = curve_fitting_effective_mass_estimates_array[optimum_effective_mass_estimate_index]

            # PLOTS
            # Plotting time dependence of Pion correlator along with the periodic exponential expression given the optimum effective mass estimate
            x_data = np.linspace(0, lattice_size, 50)
            expfunc_parameters = [ 0.5*np.min(gv.mean(jackknife_time_dependent_pion_correlator_array))*gv.exp(optimum_effective_mass_estimate*lattice_size/2.0), optimum_effective_mass_estimate]
            fig, ax = plt.subplots()
            ax.grid()
            ax.set_title('Time-dependence of the two-point Pion correlator \n(L='+str(lattice_size)+', mb='+str(bare_mass_value)+', N='+str(number_of_processed_configurations)+')', pad = 10)
            ax.set_yscale('log')
            ax.set(xlabel='$t$', ylabel='C(t)')
            plt.errorbar(range(len(jackknife_time_dependent_pion_correlator_array)), gv.mean(jackknife_time_dependent_pion_correlator_array), yerr=gv.sdev(jackknife_time_dependent_pion_correlator_array), fmt='o', markersize=8, capsize=10, label=operator_type_label.replace('_', ' ').rstrip('s')+'\n'+laplacian_stencil_label.replace('_', ' ').replace('s', 'S')+' with '+derivative_stencil_label.replace('_', ' ') )
            plt.plot(x_data, post_processing.expfunc(x_data, expfunc_parameters[0].mean, expfunc_parameters[1].mean, lattice_size), 'r-')
            ax.legend(loc="upper center")
            auxiliary.creating_directory_function(operator_plotting_directory+'/Pion_correlator/')
            fig.savefig(operator_plotting_directory+'/Pion_correlator/Time_dependence_of_the_pion_correlator_mb='+str(bare_mass_value)+'.png')
            plt.close()

            # Plotting the time-dependence of the effective mass values along with the optimal effective mass estimate
            fig, ax = plt.subplots()
            x_data = np.linspace(1, len(jackknife_effective_mass_per_time_array[1:])+1, 50)
            ax.grid()
            ax.set_title('Effective mass values as a function of time \n(L='+str(lattice_size)+', mb='+str(bare_mass_value)+', N='+str(number_of_processed_configurations)+')', pad = 10)
            ax.set(xlabel='$t$', ylabel='$m_{eff}$')
            plt.errorbar(range(1, len(jackknife_effective_mass_per_time_array[1:])+1), gv.mean(jackknife_effective_mass_per_time_array[1:]), yerr=gv.sdev(jackknife_effective_mass_per_time_array[1:]), fmt='o', markersize=8, capsize=10, label=operator_type_label.replace('_', ' ').rstrip('s')+'\n'+laplacian_stencil_label.replace('_', ' ').replace('s', 'S')+'\n with '+derivative_stencil_label.replace('_', ' ') )
            plt.plot(x_data, post_processing.plateau_fit_function(x_data, optimum_effective_mass_estimate.mean), 'r-', label='Eff. mass estimate = {:.4f}'.format(optimum_effective_mass_estimate.mean)+u"\u00B1"+'{:.4f}'.format(optimum_effective_mass_estimate.sdev)+'\n'+'Initial curve-fitting time='+str(optimum_effective_mass_estimate_index+1))
            ax.legend(loc="upper right")
            auxiliary.creating_directory_function(operator_plotting_directory+'/Effective_mass_values/')
            fig.savefig(operator_plotting_directory+'/Effective_mass_values/Time_dependence_of_the_effective_mass_mb='+str(bare_mass_value)+'.png')
            plt.close()

            # Plotting the plateau fit effective mass estimates per initial time with the corresponding chi2/dof value
            fig, ax = plt.subplots()
            ax.grid()
            ax.set(xlabel='$t_i$', ylabel='$m_{eff}^{est.}$')
            ax.set_title('Plateau fit effective mass estimates Vs. initial time \n(L='+str(lattice_size)+', mb='+str(bare_mass_value)+', N='+str(number_of_processed_configurations)+')', pad = 10)
            # Annotating the plot with the chi2/dof values positioned alternatively above and below the markers. The first initial time investigated was n_t=1.
            x = range(1, len(curve_fitting_effective_mass_estimates_array)+1)
            for i in range(len(x)):
                plt.annotate(plateau_fit_chi_square_per_dof_array[i], (x[i], gv.mean(curve_fitting_effective_mass_estimates_array[i])), xytext=(0, 30*(-1)**(i+1)), textcoords="offset pixels")
            plt.errorbar(x, gv.mean(curve_fitting_effective_mass_estimates_array), yerr=gv.sdev(curve_fitting_effective_mass_estimates_array), fmt='o', markersize=8, capsize=10, label='Index of the optimum\neffective mass estimate = '+str(optimum_effective_mass_estimate_index+1)+'\n'+operator_type_label.replace('_', ' ').rstrip('s')+'\n'+laplacian_stencil_label.replace('_', ' ').replace('s', 'S')+'\n with '+derivative_stencil_label.replace('_', ' ') )
            ax.legend(loc="upper right")
            auxiliary.creating_directory_function(operator_plotting_directory+'/Effective_mass_estimates/')
            fig.savefig(operator_plotting_directory+'/Effective_mass_estimates/Plateau_fit_effective_mass_estimates_per_initial_time_mb='+str(bare_mass_value)+'.png')
            plt.close()

            # Jackknife effective mass estimates
            jackknife_effective_mass_estimates_per_configuration = list()
            for row in range(number_of_processed_configurations):
                jackknife_effective_mass_per_time_per_configuration_array = gv.gvar(jackknife_replicas_of_effective_mass_per_time_2D_array[row], jackknife_covariance_matrix_effective_mass_per_time_array)

                # Curve-fitting datasets
                x = range(optimum_effective_mass_estimate_index, len(jackknife_effective_mass_per_time_per_configuration_array))
                y = jackknife_effective_mass_per_time_per_configuration_array[optimum_effective_mass_estimate_index:]
                # The initial estimate for the effective mass equals the value 
                p0 = [optimum_effective_mass_estimate.mean]

                fit = lsqfit.nonlinear_fit(data=(x, y), p0=p0, fcn=post_processing.plateau_fit_function, debug=True)

                jackknife_effective_mass_estimates_per_configuration.append(gv.mean(fit.p[0]))

            jackknife_effective_mass_estimates_per_configuration = np.array(jackknife_effective_mass_estimates_per_configuration)

            jackknife_effective_mass_estimates_per_configuration.tofile(output_directory+processed_files_subdirectory.replace(input_directory, "")+f'/mb={bare_mass_value:.2f}')


if __name__ == "__main__":
    main()
