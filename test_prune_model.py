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

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
    local_rank: int = -1,
):
    local_rank, world_size = setup_model_parallel()
    #local_rank, world_size = 0, 1
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")
    
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
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model = torch.load(ckpt_dir, map_location='cpu').to('cuda')
    generator = LLaMA(model, tokenizer)

    generator.model.eval()
    with torch.no_grad():
        results = generator.generate(
            prompts, max_gen_len=256, temperature=temperature, top_p=top_p, device="cuda"
        )
    
    print("\n==================Generation Results After Training================\n")
    for result in results:
        print(result)
        print("\n==================Finish================\n")
    
    print("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

    


if __name__ == "__main__":
    fire.Fire(main)
