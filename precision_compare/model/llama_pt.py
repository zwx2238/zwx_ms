import torch
from torch import nn as nn
from torch.nn import functional as F

from precision_compare.model.config import get_llama_config
from precision_compare.model.model import create_test_model
from precision_compare.utils.weight_utils import WeightManager


class LlamaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config["num_attention_heads"]
        self.hidden_size = config["hidden_size"]
        self.head_dim = self.hidden_size // self.num_attention_heads

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape

        # 投影查询、键、值
        q = (
            self.q_proj(hidden_states)
            .view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(hidden_states)
            .view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden_states)
            .view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
            .transpose(1, 2)
        )

        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=q.dtype)
        )
        attn_weights = F.softmax(scores, dim=-1)

        # 应用注意力权重
        output = torch.matmul(attn_weights, v)
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_size)
        )
        output = self.o_proj(output)

        return output


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config["hidden_size"]
        intermediate_size = config["intermediate_size"]

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = F.gelu(self.gate_proj(x))
        up = self.up_proj(x)
        intermediate = gate * up
        output = self.down_proj(intermediate)
        return output


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]

        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = nn.LayerNorm(
            self.hidden_size, eps=config["layer_norm_eps"]
        )
        self.post_attention_layernorm = nn.LayerNorm(
            self.hidden_size, eps=config["layer_norm_eps"]
        )

    def forward(self, hidden_states):
        residual = hidden_states

        # 自注意力
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        # FFN
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])

        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(config["num_hidden_layers"])]
        )
        self.norm = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.norm(hidden_states)

        return hidden_states


class LlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(
            config["hidden_size"], config["vocab_size"], bias=False
        )

    def forward(self, input_ids):
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)
        return logits

    def load_weights(self, weights_path: str, strict: bool = True):
        """加载safetensors格式的权重"""
        print(f"PyTorch模型加载权重: {weights_path}")

        # 使用WeightManager加载权重
        missing_keys = WeightManager.load_torch_weights(self, weights_path, strict)

        if not missing_keys:
            print("PyTorch模型所有权重加载成功!")
        else:
            print(f"PyTorch模型部分权重未加载，共 {len(missing_keys)} 个参数缺失")

        return missing_keys
