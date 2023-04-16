<div align="left"> <h1>LLaMA-Pruning: Structural Pruning for LLaMA</h1> </div>

<div align="left">
  <a href="https://opensource.org/licenses/GPL-3.0">
    <img src="https://img.shields.io/badge/License-GPL%203.0-4caf50.svg?style=flat-square" alt="License: GPL-3.0">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-%3E=v1.11.0-673ab7.svg?style=flat-square" alt="PyTorch>=v1.11.0">
  </a>
  <a href="https://github.com/horseee/LLaMA-Pruning/graphs/contributors">
    <img src="https://img.shields.io/github/contributors/horseee/LLaMA-Pruning.svg?style=flat-square&color=9c27b0" alt="Contributors">
  </a>
  <a href="https://github.com/VainF/Torch-Pruning">
    <img src="https://img.shields.io/badge/Torch--Pruning-v1.1.5-3f51b5.svg?style=flat-square" alt="Torch-Pruning">
  </a>
  <a href="https://github.com/facebookresearch/llama">
    <img src="https://img.shields.io/badge/LLMs-LLaMA-2196f3.svg?style=flat-square" alt="LLaMA">
  </a>
</div>

<br>

This repository provides minimal examples of pruning Large Language Models (LLMs). 

LLMs, characterized by their incredibly large number of parameters and computational demands, often present huge challenges to downstream applications. Structural Pruning offers a potential solution to this issue by removing parameters from models. To this end, this project aims to build a straightforward and general pipeline for the pruning of LLaMA and other LLMs.

**Available Features:**
- [x] [Layer Pruner](https://github.com/horseee/LLaMA-Pruning/blob/main/llama_pruner.py) for basic layers in LLaMA.
- [x] Random Pruning for LLaMA-7B.
- [x] L1/L2 Pruning for LLaMA-7B.

**TODO List:**
- [ ] Structural Pruning for LLaMA-13B/33B/65B.
- [ ] More pruners: Sailency-based Pruning.
- [ ] Code for finetuning and post-training of the pruned model.
- [ ] Quantative results.
- [ ] More LLMs.

## Qucik Start

### 0. Setup
```
pip install -r requirements.txt
```

### 1. Pretrained LLaMA
Prepare pretrained models following the [official instructions](https://github.com/facebookresearch/llama).

This example expects the following files to be available:
```
ckpt
└── LLaMA
    ├── 7B
    │   ├── checklist.chk
    │   ├── consolidated.00.pth
    │   └── params.json
    ├── tokenizer_checklist.chk
    └── tokenizer.model
```

### 2. LLaMA-7B => LLaMA-1.7B
* \#Params: 6.73B => 1.72B  
* GPU RAM: 22,067M => 7,781 M
* Requires ~20GB GPU memory on a single 3090 to prune the model. Small pruning ratio may require more GPU memory.

**Pruning:** The following script globally removes 50% of the dimensions of the LLaMA-7B model, resulting in a lightweight model with 1.72B parameters. Specify the pruner type with `--pruner_type <l1/l2/random>` and pruning ratio with `--pruning_ratio 0.5`.
```bash
python -m torch.distributed.launch --master_port 18101 --nproc_per_node 1 prune.py --pruner_type l1 --ckpt_dir ckpt/LLaMA/7B/ --tokenizer_path ckpt/LLaMA/tokenizer.model --pruning_ratio 0.5 --save_ckpt_name 'llama_prune_1.7B'
```

**Testing:** After pruning, we can load and test the pruned model with some pre-defined prompts. 
```bash
python -m torch.distributed.launch --master_port 18101 --nproc_per_node 1 test_prune_model.py --save_ckpt_name llama_prune_1.7B --tokenizer_path ckpt/LLaMA/tokenizer.model
```
Please modify the `ckpt_dir` and `tokenizer_path` according to the path to your LLaMA weights. 


### 3. Finetuning / Post Training

The pruning impacts the model's structure and hurts the performance of the model, necessitating the post-training or fine-tuning of the model on the downstream tasks. We are still developing code for this part.

## Acknowledgement

The LLaMA model is adapted from [facebookresearch/llama](https://github.com/facebookresearch/llama).  
Structural Pruning is powered by [VainF/Torch-Pruning](https://github.com/VainF/Torch-Pruning).

## Citation
```
@article{fang2023depgraph,
  title={DepGraph: Towards Any Structural Pruning},
  author={Fang, Gongfan and Ma, Xinyin and Song, Mingli and Mi, Michael Bi and Wang, Xinchao},
  journal={The IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```

```
@article{touvron2023llama,
  title={LLaMA: Open and Efficient Foundation Language Models},
  author={Touvron, Hugo and Lavril, Thibaut and Izacard, Gautier and Martinet, Xavier and Lachaux, Marie-Anne and Lacroix, Timoth{\'e}e and Rozi{\`e}re, Baptiste and Goyal, Naman and Hambro, Eric and Azhar, Faisal and Rodriguez, Aurelien and Joulin, Armand and Grave, Edouard and Lample, Guillaume},
  journal={arXiv preprint arXiv:2302.13971},
  year={2023}
}
```


