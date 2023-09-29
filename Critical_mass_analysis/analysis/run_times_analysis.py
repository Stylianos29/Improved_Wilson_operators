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
@click.option("--input_file", "input_file_path", "-in_file", default='../processing/runtimes.txt', help="The file listing the runtimes for all the completed jobs.")
@click.option("--output_directory", "output_directory", "-out_dir", default='./effective_mass_estimates/', help="The directory storing files with the analyzed effective mass estimates.")
@click.option("--plotting_directory", "plotting_directory", "-plot_dir", default='../plots/', help="The directory that will contain the analyzed plots.")

# Effective_mass_estimates

def main(input_file_path, output_directory, plotting_directory):
    
    # with open(input_file_path, 'r') as input_file:
    #     input_file_lines = input_file.readlines()
    #     for line in input_file_lines:
    #         print(line)

    temp = np.fromfile('/nvme/h/cy22sg1/Improved_Wilson_operators/Critical_mass_analysis/processed_files/KL3_overlap_operator/standard_laplacian_standard_derivative/L=16/CG=1e-06_KL_iter=3/mb=0.04_configs_total=5', dtype=np.float64)

    print(np.shape(temp))


if __name__ == "__main__":
    main()
