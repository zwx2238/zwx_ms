import os

import mindspore as ms
import numpy as np
from mindspore import nn as nn, ops as ops, Tensor
from mindspore.common import dtype as mstype

from zwx_ms.mock.mock_ms import register_module_info_ms, ms_logger
from zwx_ms.utils.weight_utils import WeightManager


class LlamaAttention(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config["num_attention_heads"]
        self.hidden_size = config["hidden_size"]
        self.head_dim = self.hidden_size // self.num_attention_heads

        self.q_proj = nn.Dense(self.hidden_size, self.hidden_size, has_bias=False)
        self.k_proj = nn.Dense(self.hidden_size, self.hidden_size, has_bias=False)
        self.v_proj = nn.Dense(self.hidden_size, self.hidden_size, has_bias=False)
        self.o_proj = nn.Dense(self.hidden_size, self.hidden_size, has_bias=False)

        # MindSpore算子
        self.softmax = nn.Softmax(axis=-1)
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.batch_matmul = ops.BatchMatMul()
        self.sqrt = ops.Sqrt()

    def construct(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape

        # 投影查询、键、值
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # 重塑和转置 - MindSpore的形状处理
        q = self.reshape(q, (batch_size, seq_len, self.num_attention_heads, self.head_dim))
        q = self.transpose(q, (0, 2, 1, 3))  # (batch_size, num_attention_heads, seq_len, head_dim)

        k = self.reshape(k, (batch_size, seq_len, self.num_attention_heads, self.head_dim))
        k = self.transpose(k, (0, 2, 1, 3))

        v = self.reshape(v, (batch_size, seq_len, self.num_attention_heads, self.head_dim))
        v = self.transpose(v, (0, 2, 1, 3))

        # 注意力计算 - MindSpore版本
        k_t = self.transpose(k, (0, 1, 3, 2))  # transpose last two dims
        scale_factor = self.sqrt(Tensor(self.head_dim, mstype.float32))
        scores = self.batch_matmul(q, k_t) / scale_factor

        attn_weights = self.softmax(scores)

        # 应用注意力权重
        output = self.batch_matmul(attn_weights, v)

        # 转置和重塑输出
        output = self.transpose(output, (0, 2, 1, 3))  # (batch_size, seq_len, num_attention_heads, head_dim)
        output = self.reshape(output, (batch_size, seq_len, self.hidden_size))

        output = self.o_proj(output)

        return output


class LlamaMLP(nn.Cell):
    def __init__(self, config):
        super().__init__()
        hidden_size = config["hidden_size"]
        intermediate_size = config["intermediate_size"]

        self.gate_proj = nn.Dense(hidden_size, intermediate_size, has_bias=False)
        self.up_proj = nn.Dense(hidden_size, intermediate_size, has_bias=False)
        self.down_proj = nn.Dense(intermediate_size, hidden_size, has_bias=False)

        # MindSpore GELU激活函数
        self.gelu = nn.GELU()
        self.mul = ops.Mul()

    def construct(self, x):
        gate = self.gelu(self.gate_proj(x))
        up = self.up_proj(x)
        intermediate = self.mul(gate, up)
        output = self.down_proj(intermediate)
        return output


class LlamaDecoderLayer(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]

        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = nn.LayerNorm((self.hidden_size,), epsilon=config["layer_norm_eps"])
        self.post_attention_layernorm = nn.LayerNorm((self.hidden_size,), epsilon=config["layer_norm_eps"])

    def construct(self, hidden_states):
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


class LlamaModel(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])

        # 使用Cell列表构建层
        self.layers = nn.CellList([LlamaDecoderLayer(config) for _ in range(config["num_hidden_layers"])])
        self.norm = nn.LayerNorm((config["hidden_size"],), epsilon=config["layer_norm_eps"])

    def construct(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.norm(hidden_states)

        return hidden_states


class LlamaForCausalLM(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.model = LlamaModel(config)
        self.lm_head = nn.Dense(config["hidden_size"], config["vocab_size"], has_bias=False)

    def construct(self, input_ids):
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)
        return logits

    def load_weights(self, weights_path: str, strict: bool = True):
        """加载safetensors格式的权重"""
        print(f"MindSpore模型加载权重: {weights_path}")

        # 使用WeightManager加载权重
        missing_keys = WeightManager.load_mindspore_weights(self, weights_path, strict)

        if not missing_keys:
            print("MindSpore模型所有权重加载成功!")
        else:
            print(f"MindSpore模型部分权重未加载，共 {len(missing_keys)} 个参数缺失")

        return missing_keys


def get_llama_config(small=True):
    """获取Llama模型配置"""
    if small:
        return {
            "vocab_size": 32000,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_hidden_layers": 2,
            "num_attention_heads": 12,
            "layer_norm_eps": 1e-5,
        }
    else:
        return {
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "layer_norm_eps": 1e-5,
        }


def create_test_model():
    """创建测试模型"""
    config = get_llama_config(small=True)
    model = LlamaForCausalLM(config)
    return model


def test_model_dump():
    """测试模型导出和记录"""
    # 设置MindSpore上下文
    ms.set_context(mode=ms.PYNATIVE_MODE)

    # 创建模型
    model = create_test_model()

    # 注册模块信息
    register_module_info_ms(model)

    # 测试输入
    batch_size = 1
    seq_len = 32
    input_ids = ms.Tensor(np.random.randint(0, 32000, (batch_size, seq_len)), ms.int32)

    # 清除已有日志
    ms_logger.clear_logs()

    # 前向传播
    outputs = model(input_ids)

    # 保存日志
    ms_logger.dump_logs("mindspore_dump_logs")

    print(f"测试日志保存到 'mindspore_dump_logs' 目录")

    return outputs


def load_model_with_weights(weights_path: str):
    """创建模型并加载权重"""
    # 设置MindSpore上下文
    ms.set_context(mode=ms.PYNATIVE_MODE)

    # 创建模型
    model = create_test_model()

    # 加载权重
    model.load_weights(weights_path)

    return model
