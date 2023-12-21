#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=44:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx8000,a100_2,a100_1,tandon_a100_2,tandon_a100_1,stake_a100_1,stake_a100_2
#SBATCH --job-name=TE_RGB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yc6317@nyu.edu
#SBATCH --output=%x.out


module purge
module load openmpi/intel/4.0.5

singularity exec --nv \
	    --bind /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
	    --overlay ../overlay_1.ext3:ro \
	    /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif \
	    /bin/bash -c "source /ext3/env.sh;         
		# mpirun python test_DDP_fast.py --model_epoch baseline_rgb_convnext_t/epoch_27 --checkpoint ./experiment/baseline_rgb_convnext_t/checkpoints/baseline_rgb_convnext_t-0.000000-0027.pth --config_file ./configs/baseline_rgb_convnext_t.yaml > ./experiment/baseline_rgb_convnext_t/eval/output_logs/03.log 2>&1 &
        # mpirun python test_DDP_fast.py --model_epoch baseline_rgb_convnext_t/epoch_34 --checkpoint ./experiment/baseline_rgb_convnext_t/checkpoints/baseline_rgb_convnext_t-0.000000-0034.pth --config_file ./configs/baseline_rgb_convnext_t.yaml > ./experiment/baseline_rgb_convnext_t/eval/output_logs/07.log 2>&1 &
		# mpirun python test_DDP_fast.py --model_epoch baseline_rgb_convnext_t/epoch_37 --checkpoint ./experiment/baseline_rgb_convnext_t/checkpoints/baseline_rgb_convnext_t-0.000000-0037.pth --config_file ./configs/baseline_rgb_convnext_t.yaml > ./experiment/baseline_rgb_convnext_t/eval/output_logs/05.log 2>&1 &
		# mpirun python test_DDP_fast_depth.py --model_epoch baseline_rgb_convnext_t/epoch_37 --checkpoint ./ --config_file ./configs/baseline_rgb_convnext_t.yaml > ./experiment/baseline_rgb_convnext_t/eval/output_logs/05.log 2>&1 &
		
		# wait

        python my_eval.py --model_name baseline_rgb_convnext_t
		"