cd experiments
cd 1
sbatch -px-men LoadB
cd ..
cd 2
sbatch -px-men LoadBL
cd ..
cd 3
sbatch -px-men LoadBT
cd ..
cd 4
sbatch -px-men LoadBLT
cd ..
cd 5
sbatch -px-men noloadBLT
cd ..

watch 'squeue | tail -n 30'
