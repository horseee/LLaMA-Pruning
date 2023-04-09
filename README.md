<div align="left"> <h1>LLaMA-Pruning: Structural Pruning for LLaMA</h1> </div>

This repository procides minimal examples of pruning Large Language Models (LLMs). LLMs, characterized by their enormous number of parameters, often present challenges related to their size and computational demands. Structural Pruning offers a potential solution to this issue by reducing the size and complexity of LLMs. 

**Available Pruners:**
* Random Pruning for LLaMA-7B

**TODO List:**
* Structural Pruning for LLaMA-13B/33B/65B
* More pruners: Magnitude-based Pruning / Sailency-based Pruning
* Finetuning and Testing.


## Qucik Start

### 0. Setup
```
pip install -r requirements.txt
```

### 1. Pretrained LLaMA
Prepare pretrained models following the [official instructions](https://github.com/facebookresearch/llama).

### 2. LLaMA-7B => LLaMA-1.7B
* \#Params: 6.73B => 1.72B  
* GPU RAM: 22,067M => 7,781 M
* Requires ~20GB GPU memory on a single 3090 to prune the model.

The instruction for pruning the model:
```bash
python -m torch.distributed.launch --master_port 18101 --nproc_per_node 1 prune.py --ckpt_dir ckpt/LLaMA/7B/ --tokenizer_path ckpt/LLaMA/tokenizer.model --pruning_ratio 0.5 --save_ckpt_name 'llama_prune_1.7B'
```

The instruction for loading and testing the pruned model:
```bash
python -m torch.distributed.launch --master_port 18101 --nproc_per_node 1 test_prune_model.py --save_ckpt_name llama_prune_1.7B --tokenizer_path ckpt/LLaMA/tokenizer.model
```

Remember to modify the `ckpt_dir` and `tokenizer_path` to be your path of storing your LLaMA.


### 3. Finetuning

We are still developing fintuning code for downstream tasks.

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


