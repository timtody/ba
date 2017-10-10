cd experiments
cd 1
sbatch -px-men experiment_1.sh
cd ..
cd 2
sbatch -px-men experiment_2.sh
cd ..
cd 3
sbatch -px-men experiment_3.sh
cd ..
cd 4
sbatch -px-men experiment_4.sh
cd ..
cd 5
sbatch -px-men experiment_5.sh
cd ..
cd 6
sbatch -px-men experiment_6.sh
cd ..






watch 'squeue | tail -n 30'
