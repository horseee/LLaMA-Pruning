# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import gc
import sys
import torch
import fire
import time
import json

import torch_pruning as tp 
from llama_pruner import RMSNormPrunner, AttentionPrunner

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA
from llama.model import RMSNorm, Attention, precompute_freqs_cis

from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
)


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    save_ckpt_name: str,
    pruning_ratio: int = 0.5,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
    local_rank: int = -1,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")
    
    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        "Building a website can be done in 10 simple steps:\n",
        # Few shot prompts: https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api
        """Tweet: "I hate it when my phone battery dies."
Sentiment: Negative
###
Tweet: "My day has been 👍"
Sentiment: Positive
###
Tweet: "This is the link to the article"
Sentiment: Neutral
###
Tweet: "This new music video was incredibile"
Sentiment:""",
        """Translate English to French:

sea otter => loutre de mer

peppermint => menthe poivrée

plush girafe => girafe peluche

cheese =>""",
    ]
    
    generator.model.eval()
    with torch.no_grad():
        results = generator.generate(
            prompts, max_gen_len=256, temperature=temperature, top_p=top_p, device="cuda"
        )
    
    print("\n==================Generation Results before Training================\n")
    for result in results:
        print(result)
    print("\n==================Finish================\n")
    
    for param in generator.model.parameters():
        param.requires_grad_(True)
    before_pruning_parameters = sum(p.numel() for p in generator.model.parameters() if p.requires_grad)
    
    example_prompts = torch.tensor([
        [    1,   306,  4658,   278,  6593,   310,  2834,   338],
        [    1,  3439, 17632,  1925, 29892,   278,  6368,   310],
    ]).cuda()

    imp = tp.importance.RandomImportance()
    
    iterative_steps = 1 
    pruner = tp.pruner.MagnitudePruner(
        generator.model,
        example_prompts,
        importance=imp,
        iterative_steps=iterative_steps,
        ch_sparsity=pruning_ratio, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        ignored_layers=[],
        customized_pruners = {
            Attention: AttentionPrunner(),
            RMSNorm: RMSNormPrunner(),
            ColumnParallelLinear: tp.pruner.function.LinearPruner(),
            RowParallelLinear: tp.pruner.function.LinearPruner(),
            ParallelEmbedding: tp.pruner.function.EmbeddingPruner()
        },
        root_module_types = [ParallelEmbedding, RMSNorm, RowParallelLinear, ColumnParallelLinear, Attention]
    )

    for i in range(iterative_steps):
        pruner.step()
        after_pruning_parameters = sum(p.numel() for p in generator.model.parameters() if p.requires_grad)
        print("#Param before: {}, #Param after: {}".format(before_pruning_parameters, after_pruning_parameters))

    # modify inferece-related attributes
    generator.model.params.dim = int(0.5 * generator.model.params.dim)
    generator.model.freqs_cis = precompute_freqs_cis(
            generator.model.params.dim // generator.model.params.n_heads, generator.model.params.max_seq_len * 2
    )

    del pruner, example_prompts
    gc.collect()
    torch.cuda.empty_cache()
    generator.model.to('cuda')
    torch.save(generator.model, os.path.join('{}.pth'.format(save_ckpt_name)))

    generator.model.eval()
    with torch.no_grad():
        results = generator.generate(
            prompts, max_gen_len=256, temperature=temperature, top_p=top_p, device="cuda"
        )
    
    print("\n==================Generation Results After Training================\n")
    for result in results:
        print(result)
        print("\n==================Finish================\n")

if __name__ == "__main__":
    fire.Fire(main)
