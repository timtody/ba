cd experiments
cd 1
sbatch -px-men LoadB
cd ..
cd 2
sbatch -px-men LoadB_wodeb
cd ..
cd 3
sbatch -px-men LoadBL
cd ..
cd 4
sbatch -px-men LoadBLT
cd ..
cd 5
sbatch -px-men loadBLT
cd ..
cd 6 sbatch -px-men LoadBK
cd ..
cd ..
watch 'squeue | tail -n 30'
