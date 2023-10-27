num_samples=10000
epochs=${1:-50}
cuda="--device cuda:0"
#cuda=""
dataset_stats="datasets/cifar-10-stats.npz"
test_exp_dir="outputs/gan/SGDA_CC/n10_f2_avg_ALIE_lr0.001_seed0_wsteps4"

for exp_dir in outputs/gan/*/*; do
    echo
    echo ${exp_dir}
    # Generate samples
    python run_gan_samples.py --use-cuda --num-samples ${num_samples} --model-path "${exp_dir}/gan_output/model_epoch${epochs}.pl"
    # Calculate FID
    python -m pytorch_fid ${cuda} ${dataset_stats} "${exp_dir}/gan_output/fake_data"
    # remove samples dir
    rm -rf "${exp_dir}/gan_output/fake_data"
done
