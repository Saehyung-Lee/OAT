# Removing Undesirable Feature Contributions using Out-of-Distribution Data

This repository is the official implementation of "Removing Undesirable Feature Contributions using Out-of-Distribution Data". 

## Requirements

To install requirements:

```setup
conda env create -f requirements.yml
```

You can download the OOD dataset and pre-trained models on CIFAR-10 here:

- [OOD dataset](https://drive.google.com/file/d/10hFYFkt6y7Mh0LpC6Tt7UN4gBd5SWaO0/view?usp=sharing)

- [Pretrained models](https://drive.google.com/file/d/1fp9P5lZZkRo3RLkqVmF2u_njU9GTR7cG/view?usp=sharing)

You can also create your own OOD datasets using the work of [Carmon et al.](https://github.com/yaircarmon/semisup-adv)

> The OAT+TRADES and OAT+RST models are trained based on their official PyTorch code.

## Training

To train the OAT model(s), run this command:

```train
python standard_train.py --oat --alpha 1.0 --suffix <model_dir_name>
or
python adversarial_train.py --oat --alpha 1.0 --suffix <model_dir_name>
```

> Please specify the path to OOD dataset in the config.json file.

## Evaluation

To evaluate the models on CIFAR-10, run:

```eval
python standard_eval.py <model_dir>
or
python adversarial_eval.py <num_steps> <xent or cw> <model_dir>
```

## Results

Our model achieves the following performance on CIFAR-10:

| Model name         | Classification Accuracy | Robustness (PGD10)     |
| ------------------ |------------------------ | ---------------------- |
| WRN-16-8           |     94.93%              |              -         |
| WRN-16-8+OAT       |     95.67%              |              -         |
| WRN-34-10+PGD      |     87.48%              |              51.93%    |
| WRN-34-10+PGD+OAT  |     86.63%              |              58.89%    |
