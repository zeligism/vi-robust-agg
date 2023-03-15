cuda="--device cuda:0"
#cuda=""
dataset_stats="datasets/cifar-10-stats.npz"
for samples_dir in outputs/gan/*/*/gan_output/fake_data; do
    #test_samples_dir="outputs/gan/SGDA_CC/n10_f2_avg_ALIE_lr0.001_seed0_wsteps4/gan_output/fake_data"
    echo ${samples_dir}
    python -m pytorch_fid ${cuda} ${dataset_stats} ${samples_dir}
done
