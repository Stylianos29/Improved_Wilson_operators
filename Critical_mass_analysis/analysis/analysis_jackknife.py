import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import gvar as gv
import lsqfit
import os
import re
import sys

# custom modules
sys.path.append('../')
import custom_library.auxiliary as auxiliary
import custom_library.post_processing as post_processing

# Data directories
input_data_directory = '../processed_files/'
# Plotting directories
output_plotting_directory = '../plots/'
auxiliary.creating_directory_function(output_plotting_directory)

run_times_dictionary = {"Improved_Wilson_operator": {"standard_laplacian_standard_derivative": '0m01', "brillouin_laplacian_isotropic_derivative": '0m04'}, "Chebyshev_overlap_operator": {"standard_laplacian_standard_derivative": '5m55', "brillouin_laplacian_isotropic_derivative": '22m18'}, "KL1_overlap_operator": {"standard_laplacian_standard_derivative": '2m015', "brillouin_laplacian_isotropic_derivative": '1m10'}, "KL2_overlap_operator": {"standard_laplacian_standard_derivative": '1m05', "brillouin_laplacian_isotropic_derivative": '1m05'}}

effective_mass_per_bare_mass_values_dictionary = {"Improved_Wilson_operator": {"standard_laplacian_standard_derivative": dict(), "brillouin_laplacian_isotropic_derivative": dict()}, "Chebyshev_overlap_operator": {"standard_laplacian_standard_derivative": dict(), "brillouin_laplacian_isotropic_derivative": dict()}, "KL1_overlap_operator": {"standard_laplacian_standard_derivative": dict(), "brillouin_laplacian_isotropic_derivative": dict()}, "KL2_overlap_operator": {"standard_laplacian_standard_derivative": dict(), "brillouin_laplacian_isotropic_derivative": dict()}}

dataset_characteristic_values_dictionary = {"Improved_Wilson_operator": {"standard_laplacian_standard_derivative": tuple(), "brillouin_laplacian_isotropic_derivative": tuple()}, "Chebyshev_overlap_operator": {"standard_laplacian_standard_derivative": tuple(), "brillouin_laplacian_isotropic_derivative": tuple()}, "KL1_overlap_operator": {"standard_laplacian_standard_derivative": tuple(), "brillouin_laplacian_isotropic_derivative": tuple()}, "KL2_overlap_operator": {"standard_laplacian_standard_derivative": tuple(), "brillouin_laplacian_isotropic_derivative": tuple()}}
# Check if corresponding directories to the operator types exist
for operator_type in effective_mass_per_bare_mass_values_dictionary.keys():
	auxiliary.creating_directory_function(output_plotting_directory+operator_type)


for subdirectory in os.listdir(os.fsencode(input_data_directory)):
	subdirectory = os.fsdecode(subdirectory)

	# Extract info from each subdirectory
	operator_type_label = re.findall(r'^.+_operator', subdirectory)[0]
	# +'s' #adding an 's' conforming with directories structure
	laplacian_stencil_label = re.findall(r'operator_(.+_laplacian)', subdirectory)[0]
	derivative_stencil_label = re.findall(r'laplacian_(.+_derivative)', subdirectory)[0]

	# Check if corresponding plotting directory to the operator specification exists
	operator_specification = laplacian_stencil_label.capitalize()+'_'+derivative_stencil_label
	operator_plotting_directory = output_plotting_directory+operator_type_label+'/'+operator_specification
	auxiliary.creating_directory_function(operator_plotting_directory)
	auxiliary.creating_directory_function(operator_plotting_directory+'/Effective_mass_estimates/')
	auxiliary.creating_directory_function(operator_plotting_directory+'/Effective_mass_values/')
	auxiliary.creating_directory_function(operator_plotting_directory+'/Pion_correlator/')

	for filename in os.listdir(os.fsencode(input_data_directory+subdirectory)):
		filename = os.fsdecode(filename)

		# Import 2D array with time-dependent Pion correlator data
		time_dependent_pion_correlator_per_configuration_2D_array = np.fromfile(input_data_directory+subdirectory+'/'+filename, dtype=np.float64)

		# Extract
		number_of_processed_configurations = int((re.findall(r'_configs_total=(\d+)', filename))[0])
		# Calculate the lattice size
		lattice_size = np.shape(time_dependent_pion_correlator_per_configuration_2D_array)[0]//number_of_processed_configurations
		# Reshape for simplifying calculation
		time_dependent_pion_correlator_per_configuration_2D_array = time_dependent_pion_correlator_per_configuration_2D_array.reshape(number_of_processed_configurations, lattice_size)
	
		dataset_characteristic_values_dictionary[operator_type_label][laplacian_stencil_label+'_'+derivative_stencil_label] = (lattice_size , number_of_processed_configurations)

		bare_mass_value = float(re.findall(r'mb=(\d*\.\d+)', filename)[0])

		# Average about its central point because the shape of the curve is expected to be symmetric due to periodic boundary conditions
		time_dependent_pion_correlator_per_configuration_2D_array = 0.5*(time_dependent_pion_correlator_per_configuration_2D_array + np.roll(np.flip(time_dependent_pion_correlator_per_configuration_2D_array, axis=1), axis = 1, shift=+1))
	
		# Jackknife calculations
		jackknife_replicas_of_time_dependent_pion_correlator_per_configuration_2D_array = post_processing.jackknife_replicas_generation(time_dependent_pion_correlator_per_configuration_2D_array)

		jackknife_average_time_dependent_pion_correlator_array = np.average(jackknife_replicas_of_time_dependent_pion_correlator_per_configuration_2D_array, axis=0)

		jackknife_variance_effective_mass_per_time_array = post_processing.jackknife_variance_array_function(jackknife_replicas_of_time_dependent_pion_correlator_per_configuration_2D_array)

		jackknife_covariance_matrix_effective_mass_per_time_array = post_processing.jackknife_covariance_matrix_function(jackknife_replicas_of_time_dependent_pion_correlator_per_configuration_2D_array)

		jackknife_time_dependent_pion_correlator_array = gv.gvar(jackknife_average_time_dependent_pion_correlator_array, jackknife_covariance_matrix_effective_mass_per_time_array)

		# For comparison purposes
		time_dependent_pion_correlator_array = (np.array([np.average(time_dependent_pion_correlator_per_configuration_2D_array, axis=0), np.std(time_dependent_pion_correlator_per_configuration_2D_array, ddof=1, axis=0)/np.sqrt(len(time_dependent_pion_correlator_per_configuration_2D_array))])).T

		# Calculate the effective mass values array using specific formula
		jackknife_replicas_of_effective_mass_per_time_2D_array = list()
		for row in range(number_of_processed_configurations):
			jackknife_replicas_of_effective_mass_per_time_2D_array.append(post_processing.effective_mass_periodic_case_function(jackknife_replicas_of_time_dependent_pion_correlator_per_configuration_2D_array[row]))
		jackknife_replicas_of_effective_mass_per_time_2D_array = np.array(jackknife_replicas_of_effective_mass_per_time_2D_array)
	
		jackknife_average_effective_mass_per_time_array = np.average(jackknife_replicas_of_effective_mass_per_time_2D_array, axis=0)

		jackknife_variance_effective_mass_per_time_array = post_processing.jackknife_variance_array_function(jackknife_replicas_of_effective_mass_per_time_2D_array)

		jackknife_covariance_matrix_effective_mass_per_time_array = post_processing.jackknife_covariance_matrix_function(jackknife_replicas_of_effective_mass_per_time_2D_array)

		jackknife_effective_mass_per_time_array = gv.gvar(jackknife_average_effective_mass_per_time_array, gv.sqrt(jackknife_variance_effective_mass_per_time_array))

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
		# METHOD 1
		shifted_curve_fitting_p_values_array = np.abs(curve_fitting_p_values_array - 0.025)
		optimum_effective_mass_estimate_index = len(shifted_curve_fitting_p_values_array) - 1 - np.argmin( np.flip(shifted_curve_fitting_p_values_array) )
		
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

		effective_mass_per_bare_mass_values_dictionary[operator_type_label][laplacian_stencil_label+'_'+derivative_stencil_label][bare_mass_value] = jackknife_effective_mass_estimates_per_configuration
	


# Plot the effective mass squared against the bare mass values plots
for operator_type in effective_mass_per_bare_mass_values_dictionary.keys():
	for operator_specification in effective_mass_per_bare_mass_values_dictionary[operator_type].keys():

		# Sort the  in an increasing fashion wrt the mass values
		jackknife_effective_mass_per_bare_mass_values_dictionary = effective_mass_per_bare_mass_values_dictionary[operator_type][operator_specification]

		if (len(jackknife_effective_mass_per_bare_mass_values_dictionary)!=0):
			
			# print(operator_type, operator_specification)

			lattice_size, number_of_processed_configurations = dataset_characteristic_values_dictionary[operator_type][operator_specification]

			run_time = run_times_dictionary[operator_type][operator_specification]

			# Sort the dictionary wrt its keys
			jackknife_effective_mass_per_bare_mass_values_dictionary = dict(sorted(jackknife_effective_mass_per_bare_mass_values_dictionary.items()))

			# Extract the sorted bare mass values
			bare_mass_values_array = np.array(list(jackknife_effective_mass_per_bare_mass_values_dictionary.keys()))

			# Extract the sorted effective mass values as 2D array
			jackknife_effective_mass_per_bare_mass_values_2D_array = np.array(list(jackknife_effective_mass_per_bare_mass_values_dictionary.values())).T

			# jackknife_effective_mass_per_bare_mass_values_array = gv.gvar(np.average(jackknife_effective_mass_per_bare_mass_values_2D_array, axis=0), (jackknife_covariance_matrix_function (jackknife_effective_mass_per_bare_mass_values_2D_array)))

			jackknife_effective_mass_per_bare_mass_values_array = gv.gvar(np.average(jackknife_effective_mass_per_bare_mass_values_2D_array, axis=0), gv.sqrt(post_processing.jackknife_variance_array_function(jackknife_effective_mass_per_bare_mass_values_2D_array)))

			# compare_two_matrices(np.diagonal((jackknife_covariance_matrix_function (jackknife_effective_mass_per_bare_mass_values_2D_array))), post_processing.jackknife_variance_array_function(jackknife_effective_mass_per_bare_mass_values_2D_array) )

			# Investigate the optimum last index
			if (len(bare_mass_values_array) >= 4):

				p_value_list = list()
				for last_index in range(3, len(bare_mass_values_array)+1):
					x = bare_mass_values_array[:last_index]
					y = (np.square(jackknife_effective_mass_per_bare_mass_values_array))[:last_index]

					# linear fit parameter guess
					slope_guess = (np.max(gv.mean(y))-np.min(gv.mean(y)))/(np.max(x)-np.min(x))
					linear_p0 = [ slope_guess, -np.min(gv.mean(y))/slope_guess ]
					linear_fit = lsqfit.nonlinear_fit(data=(x, y), p0=linear_p0, fcn=post_processing.linear_func, debug=True)
					p_value_list.append(1 - stats.chi2.cdf(linear_fit.chi2, linear_fit.dof))

				# 	# quadratic fit parameter guess
				# 	quadratic_p0 = [0.01, *linear_p0]
				# 	quadratic_p0[1] = -1.0*quadratic_p0[1]
				# 	quadratic_fit = lsqfit.nonlinear_fit(data=(x, y), p0=quadratic_p0, fcn=quadratic_func, debug=True)
				# 	# print(quadratic_fit.chi2/quadratic_fit.dof)
				# print()
				p_value_list = np.array(p_value_list)

				optimum_last_index = 4 + np.argmax(p_value_list)
			
			else:
				optimum_last_index = len(bare_mass_values_array)

			# Curve-fitting the square of the optimum effective mass estimates against the bare mass values
			if (operator_type == 'Improved_Wilson_operators'):	
				x = bare_mass_values_array[:optimum_last_index]
				y = (np.square(jackknife_effective_mass_per_bare_mass_values_array))[:optimum_last_index]
			else:
				x = bare_mass_values_array
				y = (np.square(jackknife_effective_mass_per_bare_mass_values_array))

			# linear fit parameter guess
			slope_guess = (np.max(gv.mean(y))-np.min(gv.mean(y)))/(np.max(x)-np.min(x))
			linear_p0 = [ slope_guess, -np.min(gv.mean(y))/slope_guess ]
			linear_fit = lsqfit.nonlinear_fit(data=(x, y), p0=linear_p0, fcn=post_processing.linear_func, debug=True)
			linear_fit_parameters = linear_fit.p

			# quadratic fit parameter guess
			quadratic_p0 = [0.01, *linear_p0]
			quadratic_p0[1] = -1.0*quadratic_p0[1]
			quadratic_fit = lsqfit.nonlinear_fit(data=(x, y), p0=quadratic_p0, fcn=post_processing.quadratic_func, debug=True)
			quadratic_fit_parameters = quadratic_fit.p

			def critical_kappa_value_function(critical_bare_mass_value):
				return 0.5/(critical_bare_mass_value + 4.0)
			
			# Critical mass value is defined as the value for which the effective mass value becomes effectively zero
			critical_bare_mass_value_linear = linear_fit_parameters[1]
			
			# Check which quadratic solution is closer to 0
			critical_bare_mass_value_quadratic = quadratic_fit_parameters[1]/quadratic_fit_parameters[0]
			if (gv.abs(critical_bare_mass_value_quadratic) > gv.abs(quadratic_fit_parameters[2])):
				critical_bare_mass_value_quadratic = quadratic_fit_parameters[2]

			# Plotting the optimum effective mass estimates against the bare mass values
			fig, ax = plt.subplots()
			ax.grid()
			ax.set_title('Squared effective mass for various bare mass values (L='+str(lattice_size)+', N='+str(number_of_processed_configurations)+')'+'\n['+operator_type.replace('_',' ')+' '+operator_specification.replace('_',' ')+']', pad = 6)
			ax.set(xlabel='$a m_b$', ylabel='$m^2_{eff.}$')
			# Axes lines
			ax.axhline(0, color='black') # x = 0
			ax.axvline(0, color='black') # y = 0
			plt.errorbar(bare_mass_values_array, gv.mean(jackknife_effective_mass_per_bare_mass_values_array**2), yerr=gv.sdev(jackknife_effective_mass_per_bare_mass_values_array**2), fmt='o', markersize=8, capsize=10)
			# Linear best-fit line
			x_data = np.linspace(critical_bare_mass_value_linear.mean-0.005, np.max(bare_mass_values_array)+0.005, 50)
			y_data = post_processing.linear_func(x_data, linear_fit_parameters)
			ax.plot(x_data, gv.mean(y_data), 'r-', label='$a m_c$ = {:.4f}'.format(critical_bare_mass_value_linear.mean)+u"\u00B1"+'{:.4f}'.format(critical_bare_mass_value_linear.sdev)+'\n$\kappa_c$ = {:.4f}'.format(critical_kappa_value_function(critical_bare_mass_value_linear).mean)+u"\u00B1"+'{:.4f}'.format(critical_kappa_value_function(critical_bare_mass_value_linear).sdev) \
		#    +'\nRightmost fitting index='+str(optimum_last_index)
			+'\n $χ^2$/dof={:.2f}/{:d}={:.2f}'.format(linear_fit.chi2,linear_fit.dof, linear_fit.chi2/linear_fit.dof))
			ax.fill_between(x_data, gv.mean(y_data) - gv.sdev(y_data), gv.mean(y_data) + gv.sdev(y_data), color='r', alpha=0.2)
			# Quadratic best-fit line
			x_data = np.linspace(critical_bare_mass_value_linear.mean-0.005, np.max(bare_mass_values_array)+0.005, 50)
			y_data = post_processing.quadratic_func(x_data, quadratic_fit_parameters)
			ax.plot(x_data, gv.mean(y_data), 'g-', label='$a m_c$ = {:.4f}'.format(critical_bare_mass_value_quadratic.mean)+u"\u00B1"+'{:.4f}'.format(critical_bare_mass_value_quadratic.sdev)+'\n$\kappa_c$ = {:.4f}'.format(critical_kappa_value_function(critical_bare_mass_value_quadratic).mean)+u"\u00B1"+'{:.4f}'.format(critical_kappa_value_function(critical_bare_mass_value_quadratic).sdev)+'\n$t=$'+str(run_time))
		#    +'\n $χ^2$/dof='+str(quadratic_fit.chi2/quadratic_fit.dof))
			ax.fill_between(x_data, gv.mean(y_data) - gv.sdev(y_data), gv.mean(y_data) + gv.sdev(y_data), color='g', alpha=0.2)
			ax.legend(loc="upper left")

			fig.savefig(output_plotting_directory+operator_type+'/'+operator_specification.capitalize()+'/Effective_mass_values_for_various_bare_mass_values.png')
			plt.close()

print("Done!")
