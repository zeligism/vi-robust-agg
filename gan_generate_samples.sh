num_samples=10000
epochs=50
for model_path in outputs/gan/*/*/gan_output/model_epoch${epochs}.pl; do
    #test_model_path="outputs/gan/SGDA_CC/n10_f2_avg_ALIE_lr0.001_seed0_wsteps4/gan_output/model_epoch50.pl"
    python run_gan_samples.py --use-cuda --num-samples ${num_samples} --model-path ${model_path}
done
