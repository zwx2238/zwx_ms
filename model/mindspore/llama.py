import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter
import mindspore.common.dtype as mstype
import sys
import os
import json
from pathlib import Path
import functools

# 导入基础模块路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mock import BaseOperatorLogger
from weight_utils import WeightManager

class MindSporeOperatorLogger(BaseOperatorLogger):
    """MindSpore操作日志记录器"""
    
    def log_operation(self, module, inputs, outputs):
        """记录操作"""
        # 获取模块名称
        module_name = getattr(module, '_module_name', module.__class__.__name__)
        
        # 创建模块条目
        self._create_module_entry(module_name, self._get_module_parameters(module))
        
        # 记录输入执行顺序
        self._log_execution_index(module_name, 'inputs')
        
        # 保存输入数据 - MindSpore版本
        if isinstance(inputs, (list, tuple)) and len(inputs) > 0:
            if isinstance(inputs[0], Tensor):
                # 转换Tensor为numpy或其他可保存格式
                self.logs[module_name]['inputs'].append(inputs[0].asnumpy())
            else:
                # 处理复杂输入
                processed_inputs = []
                for x in inputs:
                    if isinstance(x, Tensor):
                        processed_inputs.append(x.asnumpy())
                    else:
                        processed_inputs.append(x)
                self.logs[module_name]['inputs'].append(processed_inputs)
        
        # 记录输出执行顺序
        self._log_execution_index(module_name, 'outputs')
            
        # 保存输出数据 - MindSpore版本
        if isinstance(outputs, Tensor):
            self.logs[module_name]['outputs'].append(outputs.asnumpy())
        elif isinstance(outputs, (tuple, list)):
            # 处理复杂输出
            processed_outputs = []
            for x in outputs:
                if isinstance(x, Tensor):
                    processed_outputs.append(x.asnumpy())
                else:
                    processed_outputs.append(x)
            self.logs[module_name]['outputs'].append(processed_outputs)
    
    def _get_module_parameters(self, module) -> dict:
        """获取模块的所有参数 - MindSpore版本"""
        params = {}
        for name in module._params:
            param = getattr(module, name)
            if isinstance(param, Parameter):
                params[name] = param.asnumpy()
        return params
    
    def _save_module_data(self, path, data):
        """保存模块数据到文件系统 - MindSpore版本"""
        # 保存输入数据
        if data['inputs']:
            input_data_path = path / "inputs.npy"
            np.save(str(input_data_path), data['inputs'])
        
        # 保存输出数据
        if data['outputs']:
            output_data_path = path / "outputs.npy"
            np.save(str(output_data_path), data['outputs'])
        
        # 分别保存每个参数，只使用参数名
        if data['parameters']:
            for param_name, param_array in data['parameters'].items():
                param_path = path / f"{param_name}.npy"
                np.save(str(param_path), param_array)

# 创建全局记录器实例
ms_logger = MindSporeOperatorLogger()

def wrap_construct(module):
    """包装模块的construct方法 - MindSpore版本"""
    original_construct = module.construct
    
    @functools.wraps(original_construct)
    def wrapped_construct(*args, **kwargs):
        outputs = original_construct(*args, **kwargs)
        ms_logger.log_operation(module, args, outputs)
        return outputs

    # 安全地获取模块名称
    module_name = getattr(module, '_module_name', module.__class__.__name__)
    # print(f"包装模块: {module_name}")
    
    module.construct = wrapped_construct

def register_module_info(module, prefix: str = ''):
    """为模块注册名称信息并包装construct方法 - MindSpore版本"""
    # 设置当前模块的名称
    module._module_name = prefix if prefix else module.__class__.__name__
    
    # 包装construct方法
    wrap_construct(module)
    
    # 递归处理子模块 - MindSpore版本
    for name, child in module._cells.items():
        if child is not None:
            full_name = f"{prefix}.{name}" if prefix else name
            register_module_info(child, full_name)

# Llama模型的MindSpore实现
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
    register_module_info(model)
    
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

if __name__ == "__main__":
    # 测试权重加载
    output_dir = "weights"
    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, "llama_shared.safetensors")
    
    # 检查权重文件是否存在
    if not os.path.exists(weights_path):
        print(f"警告: 权重文件不存在: {weights_path}")
        print("将使用PyTorch端生成权重文件...")
        
        # 如果权重文件不存在，先使用PyTorch生成
        from model.torch.llama import get_llama_config as get_torch_config
        WeightManager.generate_shared_weights(get_torch_config(small=True), weights_path)
    
    # 创建MindSpore模型并加载权重
    model = load_model_with_weights(weights_path)
    
    # 测试模型
    inputs = ms.Tensor(np.random.randint(0, 32000, (1, 10)), ms.int32)
    outputs = model(inputs)
    
    print(f"模型输出形状: {outputs.shape}")
    print("MindSpore模型权重加载测试完成!") 