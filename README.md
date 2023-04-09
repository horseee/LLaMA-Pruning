<div align="left"> <h1>LLaMA-Pruning: Structural Pruning for LLaMa</h1> </div>

This repsository is intended as a intial exploration for pruning Large Language Models (LLMs). 

## 0. Setup
```
pip install -r requirements.txt
```

## 1. Download Pretrained Models
Following instructions from [the official repo](https://github.com/facebookresearch/llama).

## 2. Pruning and Testing (without Re-training)
```python
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --master_port 18100 --nproc_per_node 1 prune_llama.py --ckpt_dir ckpt/LLaMa/7B/ --tokenizer_path ckpt/LLaMa/tokenizer.model
```
  
### Outputs of Pruned LLaMA-7B

<hr>
  
* **Prompt 1:**
  "I believe the meaning of life is",

* **Outputs 1 (Before Pruning):**
  
  I believe the meaning of life is to find happiness and be satisfied with what you have. 
  People have different definitions of happiness. Some people feel that if they could only 
  win the lottery, they would be happy. Some people feel that if they could only get that promotion, 
  they would be happy. Some people feel that if they could only be the top scorer in a game, they would be happy.
  If you do not know what happiness is, I suggest you ask a psychologist. 
  A psychologist has studied the subject of happiness and he or she knows what happiness is. 
  A psychologist has a Ph.D. in psychology and is a licensed psychologist.
  Do not waste your money on a psychic who claims to be able to read your mind and tell you 
  what you should do with your life. I have seen many psychics, and I have never met one who could read my mind. 
  The psychics are all just wasting your money.
  Many people think that the meaning of life is to be successful in everything. 
  But, I believe that the meaning of life is to be content with what you have. I believe that happiness is the meaning of life.
  I used to think that the meaning of life was to be successful in everything. I was

* **Outputs 2 (After Pruning):**
  
<hr>


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
