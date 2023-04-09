<div align="center"> <h1>LLaMA-Pruning <br> <h3>Structural Pruning for LLaMa<h3> </h1> </div>

This repsository is intended as a intial exploration for pruning Large Language Models (LLMs). 

## Setup
```
pip install -r requirements.txt
```

## Download Pretrained Models
Following instructions from [the official repo](https://github.com/facebookresearch/llama).

## Pruning and Test
```python
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --master_port 18100 --nproc_per_node 1 prune_llama.py --ckpt_dir ckpt/LLaMa/7B/ --tokenizer_path ckpt/LLaMa/tokenizer.model
```

## Finetuning

We are still developing fintuning code for downstream tasks.

## Acknowledgement

The LLaMA model is adapted from [facebookresearch/llama](https://github.com/facebookresearch/llama)
Pruning is powered by [VainF/Torch-Pruning](https://github.com/VainF/Torch-Pruning)

## Citation
```
@article{fang2023depgraph,
  title={DepGraph: Towards Any Structural Pruning},
  author={Fang, Gongfan and Ma, Xinyin and Song, Mingli and Mi, Michael Bi and Wang, Xinchao},
  journal={The IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```
