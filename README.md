# VI-Robust-Aggregator

This is the official implementation of the Adversarial MNIST experiments and the CIFAR-10 GAN experiments in [Byzantine-Tolerant Methods for Distributed Variational Inequalities](https://openreview.net/forum?id=ER0bcYXvvo).


| ![FID_NA](assets/FID_NA.png) | ![FID_LF](assets/FID_LF.png) |
|-|-|
| ![FID_IPM](assets/FID_IPM.png) |  ![FID_ALIE](assets/FID_ALIE.png) |

## Requirements
Standard deep learning libraries, such as NumPy, Matplotlib, PyTorch, etc. Try this command:
```
    conda create -n NAMEOFMYENV pytorch torchvision torchaudio matplotlib numpy scikit-learn -c pytorch
```

## Training
Code for running all the experiments is indicated with a `run_` prefix.

Particularly, Adversarial MNIST experiments can be run with
```
    python run_adv_experiments.py --adversarial --use-cuda
```
And CIFAR-10 GAN experiments can be run with
```
    python run_experiments.py --gan --use-cuda
```

We also provide job scripts for running the experiments on SLURM, which can be easily adjusted for your compute cluster. There are also other scripts for running other experiments, but they can be safely ignored.

### Some extra details
The backbone code was taken from this [repo](https://github.com/epfml/byzantine-robust-noniid-optimizer). The main script is (unfortunately) called `utils.py`, which has been heavily changed from the original. The experiments can be run by calling the `main` function from this file, as can be seen from the `run_*.py` scripts.

