# Removing Undesirable Feature Contributions using Out-of-Distribution Data

This repository is the official implementation of "Removing Undesirable Feature Contributions using Out-of-Distribution Data", published as a conference paper at [ICLR 2021](https://openreview.net/forum?id=eIHYL6fpbkA).


## Requirements

- Python (3.6.4)
- Pytorch (0.4.1)
- CUDA
- numpy

You can download the OOD dataset and pre-trained models on CIFAR-10 here:

- [OOD dataset](https://drive.google.com/file/d/13Nyw3b8lBfBTbVnUEw_yyFGW7x6rjWRD/view?usp=sharing)

- [OAT+TRADES](https://drive.google.com/file/d/1p7UEBeVjQfu3W5CWzhvkUga5iDxGjDTL/view?usp=sharing)

You can also create your own OOD datasets using the work of [Carmon et al.](https://github.com/yaircarmon/semisup-adv)

## Training

To train the OAT model(s), run this command:

```train
python train_trades_cifar10.py --oat --aug-dataset-dir <path_to_OOD> --model-dir <model_dir_name>
```

## Evaluation

To evaluate the models on CIFAR-10, run:

```eval
python pgd_attack_cifar10.py --model-path <path_to_model>
```
