import torch
import torch.nn as nn
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from vllm.model_executor.parallel_utils import parallel_state
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size)
from vllm.attention import Attention, AttentionMetadata
# from vllm import _custom_ops as ops
from vllm._C import cache_ops, ops

### test setting ###
Z = 16
DIM = 5120           # hidden_size(Llama2-7B)
LEN = 16           # seq_len, use LEN-1 context to infer the LEN-th token
MAX_LEN = 1024      # max length of KV cache
HN = 40              # num_heads in attn
HS = 128             # head_dim in attn (hidden_size / head_num)


class RotaryEmbedding(nn.Module):
    """Original rotary positional embedding."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style

        cache = self._compute_cos_sin_cache()
        cache = cache.half()
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        """Compute the inverse frequency."""
        # NOTE(woosuk): The HF implementation uses `torch.arange(...).float()`.
        # However, we use `torch.arange(..., dtype=torch.float)` instead to
        # avoid numerical issues with large base values (e.g., 10000000).
        # This may cause a slight numerical difference between the HF
        # implementation and ours.
        # NOTE(woosuk): To exactly match the HF implementation, we need to
        # use CPU to compute the cache and then move it to GPU. However, we
        # create the cache on GPU for faster initialization. This may cause
        # a slight numerical difference between the HF implementation and ours.
        inv_freq = 1.0 / (base**(torch.arange(
            0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def _forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        query = query.view(*query.shape[:-1], -1, self.head_size)
        key = key.view(*key.shape[:-1], -1, self.head_size)

        query_rot = query[..., :self.rotary_dim]
        key_rot = key[..., :self.rotary_dim]
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim:]
            key_pass = key[..., self.rotary_dim:]

        self.cos_sin_cache = self.cos_sin_cache.to(positions.device)
        cos_sin = self.cos_sin_cache[torch.add(positions, offsets)
                                     if offsets is not None else positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_neox_style:
            # NOTE(woosuk): Here we assume that the positions tensor has the
            # shape [batch_size, seq_len].
            cos = cos.repeat(1, 1, 2).unsqueeze(-2)
            sin = sin.repeat(1, 1, 2).unsqueeze(-2)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

        rotate_fn = _rotate_neox if self.is_neox_style else _rotate_gptj
        query_rot = query_rot * cos + rotate_fn(query_rot) * sin
        key_rot = key_rot * cos + rotate_fn(key_rot) * sin

        if self.rotary_dim < self.head_size:
            query = torch.cat((query_rot, query_pass), dim=-1)
            key = torch.cat((key_rot, key_pass), dim=-1)
        else:
            query = query_rot
            key = key_rot
        query = query.flatten(-2)
        key = key.flatten(-2)
        return query, key

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.cos_sin_cache = self.cos_sin_cache.to(positions.device)
        # ops.rotary_embedding()/batched_rotary_embedding()
        # are in-place operations that update the query and key tensors.
        if offsets is not None:
            ops.batched_rotary_embedding(positions, query, key, self.head_size,
                                         self.cos_sin_cache,
                                         self.is_neox_style, self.rotary_dim,
                                         offsets)
        else:
            ops.rotary_embedding(positions, query, key, self.head_size,
                                 self.cos_sin_cache, self.is_neox_style)
        return query, key

# tp_size = get_tensor_model_parallel_world_size()
torch.distributed.init_process_group(
    backend="nccl",
    world_size=1,
    rank=0,
    init_method="tcp://localhost:8088",
)
parallel_state.initialize_model_parallel(1, 1)
hidden_states = torch.empty((Z, 1, DIM), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)  
qkv_proj = QKVParallelLinear(
            DIM,
            HS,
            HN,
            HN,
            bias=False).half().cuda()
# qkv_proj = nn.Linear(DIM, DIM * 3, dtype=torch.float16, device="cuda")

rotary_emb = RotaryEmbedding(head_size=HS, rotary_dim=HS, max_position_embeddings=MAX_LEN, base=10000, is_neox_style=False)
positions = torch.zeros((Z, 1), dtype=torch.int64, device="cuda")

slot_mapping = torch.arange(start=0, end=Z, step=1, dtype=torch.int64, device="cuda")

k_cache = torch.empty((1024, HN, HS // 8, 16, 8), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)  
v_cache = torch.empty((1024, HN, HS, 16), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5) 

####################
# begin test part
def part_test(x):
    qkv, _ = qkv_proj(x)
    q, k, v = qkv.split([DIM, DIM, DIM], dim=-1)
    q, k = rotary_emb(positions, q, k)

    k = k.view(-1, HN, HS)
    v = v.view(-1, HN, HS) 

    cache_ops.reshape_and_cache(
            k,
            v,
            k_cache,
            v_cache,
            slot_mapping.flatten(),
            'auto',
        )

WARM = 10
FREQ = 100

for _ in range(WARM):
    part_test(hidden_states)

start_event = [torch.cuda.Event(enable_timing=True) for i in range(FREQ)]
end_event = [torch.cuda.Event(enable_timing=True) for i in range(FREQ)]
for i in range(FREQ):
    start_event[i].record()
    part_test(hidden_states)
    end_event[i].record()
torch.cuda.synchronize()
dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)


print("%.4f" % torch.mean(dur).item())



