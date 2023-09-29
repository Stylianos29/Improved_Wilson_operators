#!/bin/bash -l
#SBATCH --job-name=CritMass
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=64
#SBATCH --time=24:00:00
#SBATCH --partition=p100
#SBATCH --error=./jobs_info/run.err
#SBATCH --output=./jobs_info/run.txt

module purge
module load gompi
module load Anaconda3
module load h5py

# Arguments:
# 1 - number of processes
# 2 - initial index
# 3 - bare mass value

# time mpirun -n $1 python pion_correlators_parallel.py -d_idx 0 -l_idx 0 --overlap_boolean True -in_idx $2 -mb $3 -CG_p '1E-6' -NSF 50 --input_file '/nvme/h/cy22sg1/Improved_Wilson_operators/u1-hmc/out_L=16.h5'

for ((bare_mass_value = 1 ; bare_mass_value < 6 ; bare_mass_value++)); do
    time python pion_correlators_serial.py --input_file '/nvme/h/cy22sg1/Improved_Wilson_operators/u1-hmc/out_L=16.h5' --bare_mass_value '0.0'"$bare_mass_value" --derivative_index 0 --laplacian_index 0 --overlap_boolean True --sign_function_boolean False --KL_iterations 3
done

# time python pion_correlators_serial.py -mb 0.02 --input_file '../../u1-hmc/out_L=24.h5'
# time python pion_correlators_serial.py -mb 0.03 --input_file '../../u1-hmc/out_L=24.h5'
# time python pion_correlators_serial.py -mb 0.04 --input_file '../../u1-hmc/out_L=24.h5'
# time python pion_correlators_serial.py -mb 0.05 --input_file '../../u1-hmc/out_L=24.h5'