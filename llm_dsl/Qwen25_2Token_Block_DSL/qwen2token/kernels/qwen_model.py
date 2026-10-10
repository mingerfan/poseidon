"""Qwen2.5 0.5B reference DSL with explicit prefill/decode KV state.

Defaults mirror the supplied my_qwen2.py. Inputs are encrypted embeddings;
weights and polynomial approximation profiles are public compile-time data.
"""

from dataclasses import dataclass
import math

import numpy as np

from . import qwen_layers as ql


@dataclass(frozen=True)
class QwenConfig:
    num_layers: int = 24
    hidden_size: int = 896
    intermediate_size: int = 4864
    query_heads: int = 14
    kv_heads: int = 2
    head_dim: int = 64
    vocab_size: int = 151936
    max_sequence_length: int = 512
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6

    def __post_init__(self):
        for name in ("num_layers", "hidden_size", "intermediate_size", "query_heads",
                     "kv_heads", "head_dim", "vocab_size", "max_sequence_length"):
            ql._integer(getattr(self, name), name)
        if self.hidden_size != self.query_heads * self.head_dim:
            raise ValueError("hidden_size must equal query_heads * head_dim")
        if self.query_heads % self.kv_heads or self.head_dim % 2:
            raise ValueError("Q heads must be divisible by KV heads, and head_dim must be even")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("rope_theta must be finite and positive")
        if not math.isfinite(self.rms_norm_eps) or self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be finite and positive")


def weight_shapes(config):
    """Names match my_qwen2.py state_dict; no implicit HF aliases or tying."""
    hidden, ffn = config.hidden_size, config.intermediate_size
    kv = config.kv_heads * config.head_dim
    shapes = {
        "emb_weight.weight": (config.vocab_size, hidden),
        "final_normal.weight": (hidden,),
        "lm_head.weight": (config.vocab_size, hidden),
    }
    for index in range(config.num_layers):
        prefix = f"model_list.{index}."
        shapes.update({prefix + name: shape for name, shape in {
            "pre_Normal.weight": (hidden,),
            "post_Normal.weight": (hidden,),
            "attention.q_weight.weight": (hidden, hidden),
            "attention.q_weight.bias": (hidden,),
            "attention.k_weight.weight": (kv, hidden),
            "attention.k_weight.bias": (kv,),
            "attention.v_weight.weight": (kv, hidden),
            "attention.v_weight.bias": (kv,),
            "attention.o_weight.weight": (hidden, hidden),
            "mlplayer.gate_proj.weight": (ffn, hidden),
            "mlplayer.up_proj.weight": (ffn, hidden),
            "mlplayer.down_proj.weight": (hidden, ffn),
        }.items()})
    return shapes


@dataclass(frozen=True)
class BlockApproximations:
    pre_norm: ql.PolynomialApproximation | ql.InverseSqrtApproximation | ql.ChebyshevApproximation
    post_norm: ql.PolynomialApproximation | ql.InverseSqrtApproximation | ql.ChebyshevApproximation
    silu: ql.PolynomialApproximation | ql.ChebyshevApproximation
    exp: ql.PolynomialApproximation | None = None
    reciprocal: ql.PolynomialApproximation | None = None
    public_shift: float = 0.0
    softmax: ql.NormalizeSquareSoftmax | None = None

    def __post_init__(self):
        names = [("pre_norm", "rsqrt"), ("post_norm", "rsqrt"), ("silu", "silu")]
        if self.softmax is None:
            names += [("exp", "exp"), ("reciprocal", "reciprocal")]
        elif (not isinstance(self.softmax, ql.NormalizeSquareSoftmax) or self.exp is not None or
              self.reciprocal is not None or self.public_shift != 0):
            raise ValueError("choose one softmax approximation method")
        for name, function in names:
            if getattr(self, name) is None:
                raise ValueError(f"missing {name} approximation")
            if getattr(self, name).function != function:
                raise ValueError(f"{name} requires a {function} approximation")
        if not math.isfinite(self.public_shift):
            raise ValueError("public_shift must be finite")


@dataclass(frozen=True)
class ModelApproximations:
    blocks: tuple
    final_norm: ql.PolynomialApproximation | ql.InverseSqrtApproximation | ql.ChebyshevApproximation

    def __post_init__(self):
        if self.final_norm.function != "rsqrt":
            raise ValueError("final_norm requires a rsqrt polynomial")
        if any(not isinstance(block, BlockApproximations) for block in self.blocks):
            raise TypeError("blocks must contain BlockApproximations")
        object.__setattr__(self, "blocks", tuple(self.blocks))

    def to_dict(self):
        """Include method tags so Chebyshev coefficients cannot become powers."""
        blocks = []
        for block in self.blocks:
            names = ["pre_norm", "post_norm", "silu"]
            names += ["softmax"] if block.softmax is not None else ["exp", "reciprocal"]
            row = {name: ql.approximation_to_dict(getattr(block, name)) for name in names}
            if block.softmax is None:
                row["public_shift"] = block.public_shift
            blocks.append(row)
        return {"blocks": blocks, "final_norm": ql.approximation_to_dict(self.final_norm)}

    @classmethod
    def from_dict(cls, data):
        if set(data) != {"blocks", "final_norm"}:
            raise ValueError("approximation JSON requires exactly blocks and final_norm")
        blocks = []
        for entry in data["blocks"]:
            # dataclasses.asdict of a legacy block includes the unused field.
            entry = {k: v for k, v in entry.items() if not (k == "softmax" and v is None)}
            required = {"pre_norm", "post_norm", "silu"}
            required |= {"softmax"} if "softmax" in entry else {"exp", "reciprocal"}
            optional = set() if "softmax" in entry else {"public_shift"}
            if not required <= set(entry) or set(entry) - required - optional:
                raise ValueError("each block requires norms, silu, and exactly one softmax method")
            approximations = {name: ql.approximation_from_dict(entry[name]) for name in required}
            blocks.append(BlockApproximations(**approximations, public_shift=entry.get("public_shift", 0.0)))
        return cls(tuple(blocks), ql.approximation_from_dict(data["final_norm"]))


@dataclass(frozen=True)
class LayerKVCache:
    """K is already RoPE-transformed. V is unrotated. One vector per position."""
    positions: tuple
    keys: tuple
    values: tuple

    def __post_init__(self):
        positions = ql._positions(self.positions, len(self.keys), "cache positions")
        if len(self.keys) != len(self.values):
            raise ValueError("cache K/V counts must match")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "keys", tuple(self.keys))
        object.__setattr__(self, "values", tuple(self.values))


@dataclass(frozen=True)
class ModelOutput:
    logits: tuple
    cache: tuple


class Qwen25Model:
    def __init__(self, weights, approximations, *, slots, logit_chunk_size,
                 config=QwenConfig()):
        self.config = config
        self.slots = ql._integer(slots, "slots")
        self.logit_chunk_size = ql._integer(logit_chunk_size, "logit_chunk_size")
        if slots & (slots - 1) or max(config.hidden_size, config.intermediate_size) > slots:
            raise ValueError("slots must be a power of two fitting hidden and FFN vectors")
        if self.logit_chunk_size > slots:
            raise ValueError("logit_chunk_size must fit slots")
        if len(approximations.blocks) != config.num_layers:
            raise ValueError("one explicit approximation profile is required per decoder block")
        self.approximations = approximations
        self.weights = {}
        for name, shape in weight_shapes(config).items():
            if name not in weights:
                raise ValueError(f"missing model weight: {name}")
            value = np.asarray(weights[name])
            if value.shape != shape or value.dtype.kind not in "iuf":
                raise ValueError(f"{name} must be a real array with shape {shape}, got {value.shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains nonfinite values")
            self.weights[name] = value

    def client_embedding(self, token_ids):
        """Client plaintext lookup; caller encrypts rows before prefill/decode."""
        return ql.client_embedding(token_ids, self.weights["emb_weight.weight"])

    def prefill(self, embeddings, *, positions):
        """Fresh causal sequence; returns last-token logits and all layer KV."""
        cache = tuple(LayerKVCache((), (), ()) for _ in range(self.config.num_layers))
        return self._forward(tuple(embeddings), positions, cache)

    def decode(self, embedding, cache, *, position):
        """Append exactly one token to nonempty KV cache; cache is not mutated."""
        cache = tuple(cache)
        if not cache or not cache[0].positions:
            raise ValueError("decode requires nonempty cache from prefill or prior decode")
        return self._forward((embedding,), (position,), cache)

    def _validate_inputs(self, embeddings, positions, cache):
        cfg = self.config
        if not embeddings:
            raise ValueError("at least one encrypted embedding is required")
        positions = ql._positions(positions, len(embeddings), "positions")
        if positions[-1] >= cfg.max_sequence_length:
            raise ValueError("position exceeds the configured context limit")
        if any(x.size != cfg.hidden_size or x.slots != self.slots for x in embeddings):
            raise ValueError("embedding shape/slot capacity does not match model")
        if len(cache) != cfg.num_layers:
            raise ValueError("cache must have one entry per decoder block")
        old_positions = cache[0].positions
        if old_positions and positions[0] <= old_positions[-1]:
            raise ValueError("new positions must follow all cached positions")
        if len(old_positions) + len(positions) > min(cfg.max_sequence_length, self.slots):
            raise ValueError("attention length exceeds context or packed score capacity")
        for entry in cache:
            if entry.positions != old_positions:
                raise ValueError("all decoder caches must describe the same positions")
            if any(x.size != cfg.kv_heads * cfg.head_dim or x.slots != self.slots
                   for x in entry.keys + entry.values):
                raise ValueError("cached K/V shape or slot capacity does not match model")
        return positions

    def _decoder_block(self, hidden, positions, cache, index):
        cfg = self.config
        profile = self.approximations.blocks[index]
        prefix = f"model_list.{index}."

        def weight(name):
            return self.weights[prefix + name]

        normalized = tuple(ql.rms_norm(x, weight("pre_Normal.weight"),
                                      rsqrt=profile.pre_norm, eps=cfg.rms_norm_eps) for x in hidden)
        attention, keys, values = ql.attention(
            normalized, weight("attention.q_weight.weight"), weight("attention.k_weight.weight"),
            weight("attention.v_weight.weight"), weight("attention.o_weight.weight"), positions=positions,
            q_bias=weight("attention.q_weight.bias"), k_bias=weight("attention.k_weight.bias"),
            v_bias=weight("attention.v_weight.bias"), cached_positions=cache.positions,
            cached_keys=cache.keys, cached_values=cache.values, query_heads=cfg.query_heads,
            kv_heads=cfg.kv_heads, head_dim=cfg.head_dim, theta=cfg.rope_theta,
            exp=profile.exp, reciprocal=profile.reciprocal, public_shift=profile.public_shift,
            normalization=profile.softmax,
        )
        updated = LayerKVCache(cache.positions + positions, keys, values)
        output = []
        for residual, attended in zip(hidden, attention):
            x = ql.add(residual, attended)
            normalized = ql.rms_norm(x, weight("post_Normal.weight"),
                                     rsqrt=profile.post_norm, eps=cfg.rms_norm_eps)
            ffn = ql.swiglu(normalized, weight("mlplayer.gate_proj.weight"),
                            weight("mlplayer.up_proj.weight"), weight("mlplayer.down_proj.weight"),
                            silu=profile.silu)
            output.append(ql.add(x, ffn))
        return tuple(output), updated

    def _forward(self, embeddings, positions, cache):
        positions = self._validate_inputs(embeddings, positions, cache)
        hidden = embeddings
        updated = []
        for index in range(self.config.num_layers):
            hidden, entry = self._decoder_block(hidden, positions, cache[index], index)
            updated.append(entry)
        # The reference model returns logits only for the last input token.
        last = ql.rms_norm(hidden[-1], self.weights["final_normal.weight"],
                          rsqrt=self.approximations.final_norm, eps=self.config.rms_norm_eps)
        logits = ql.lm_head(last, self.weights["lm_head.weight"], chunk_size=self.logit_chunk_size)
        return ModelOutput(logits, tuple(updated))
