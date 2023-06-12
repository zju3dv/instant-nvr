# Learning Neural Volumetric Representations of Dynamic Humans in Minutes

### [Project Page](https://zju3dv.github.io/instant_nvr) | [Video](https://zju3dv.github.io/instant_nvr) | [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Geng_Learning_Neural_Volumetric_Representations_of_Dynamic_Humans_in_Minutes_CVPR_2023_paper.pdf) | [Data](https://github.com/zju3dv/instant_nvr)

![inb](https://chen-geng.com/instant_nvr/images/inb.gif)

> [Learning Neural Volumetric Representations of Dynamic Humans in Minutes](https://zju3dv.github.io/instant_nvr)
>
> Chen Geng\*, Sida Peng\*, Zhen Xu\*, Hujun Bao, Xiaowei Zhou (* denotes equal contribution)
>
> CVPR 2023

## Installation

See [here](./docs/install.md).

## Reproducing results in the paper

We provide two scripts to help reproduce the results shown in the paper.

After installing the environment and the dataset, for evaluation on the ZJU-MoCap dataset, run:

```shell
sh scripts/eval_zjumocap.sh
```

For evaluation on the MonoCap dataset, run:

```shell
sh scripts/eval_monocap.sh
```


## Evaluation on ZJU-MoCap

Taking 377 as an example.

Training on ZJU-MoCap.

```shell
export name=377
python train_net.py --cfg_file configs/inb/inb_${name}.yaml exp_name inb_${name} gpus ${GPUS} silent True
```

Evaluation:
```shell
export name=377
python run.py --type evaluate --cfg_file configs/inb/inb_${name}.yaml exp_name inb_${name} gpus ${GPUS}
```

## Evaluation on MonoCap

Taking "lan" as an example.

Training on Monocap. 

```shell
export name=lan
python train_net.py --cfg_file configs/inb/inb_${name}.yaml exp_name inb_${name} gpus ${GPUS} silent True
```

Evaluation:
```shell
export name=lan
python run.py --type evaluate --cfg_file configs/inb/inb_${name}.yaml exp_name inb_${name} gpus ${GPUS}
```

## TODO List

This repository currently serves as the release of the technical paper's implementation and will undergo future updates (planned below) to enhance user-friendliness. We warmly welcome and appreciate any contributions.

- [ ] Instruction on running on custom datasets
- [ ] Add support for further acceleration using CUDA
- [ ] Add a Google Colab notebook demo

## Bibtex

If you find the repo useful for your research, please consider citing our paper:

```
@inproceedings{instant_nvr,
    title={Learning Neural Volumetric Representations of Dynamic Humans in Minutes},
    author={Chen Geng and Sida Peng and Zhen Xu and Hujun Bao and Xiaowei Zhou},
    booktitle={CVPR},
    year={2023}
}
```