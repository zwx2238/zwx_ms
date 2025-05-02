import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# 导入基础模块路径
from zwx_ms.mock.mock import BaseOperatorLogger
from zwx_ms.utils.weight_utils import WeightManager

class TorchOperatorLogger(BaseOperatorLogger):
    """PyTorch操作日志记录器"""
    
    def log_operation(self, module: torch.nn.Module, inputs: tuple, outputs: torch.Tensor):
        """记录操作"""
        # 获取模块名称
        module_name = getattr(module, '_module_name', module.__class__.__name__)
        
        # 创建模块条目
        self._create_module_entry(module_name, self._get_module_parameters(module))
        
        # 记录输入执行顺序
        self._log_execution_index(module_name, 'inputs')
        
        # 保存输入数据
        if isinstance(inputs[0], torch.Tensor):
            self.logs[module_name]['inputs'].append(inputs[0].detach().cpu())
        elif isinstance(inputs, (tuple, list)) and len(inputs) > 0:
            self.logs[module_name]['inputs'].append([x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in inputs])
        
        # 记录输出执行顺序
        self._log_execution_index(module_name, 'outputs')
            
        # 保存输出数据
        if isinstance(outputs, torch.Tensor):
            self.logs[module_name]['outputs'].append(outputs.detach().cpu())
        elif isinstance(outputs, (tuple, list)):
            self.logs[module_name]['outputs'].append([x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in outputs])
    
    def _get_module_parameters(self, module: torch.nn.Module) -> dict:
        """获取模块的所有参数"""
        params = {}
        for name, param in module.named_parameters(recurse=False):  # 不递归获取子模块的参数
            if param.requires_grad:  # 只保存需要梯度的参数
                params[name] = param.detach().cpu()
        return params
    
    def _save_module_data(self, path, data):
        """保存模块数据到文件系统"""
        # 保存输入数据
        if data['inputs']:
            input_data_path = path / "inputs.pt"
            torch.save(data['inputs'], str(input_data_path))
        
        # 保存输出数据
        if data['outputs']:
            output_data_path = path / "outputs.pt"
            torch.save(data['outputs'], str(output_data_path))
        
        # 分别保存每个参数，只使用参数名
        if data['parameters']:
            for param_name, param_tensor in data['parameters'].items():
                param_path = path / f"{param_name}.pt"
                torch.save(param_tensor, str(param_path))

# 创建全局记录器实例
torch_logger = TorchOperatorLogger()

def wrap_forward(module: torch.nn.Module):
    """包装模块的forward方法"""
    original_forward = module.forward
    
    @functools.wraps(original_forward)
    def wrapped_forward(*args, **kwargs):
        outputs = original_forward(*args, **kwargs)
        torch_logger.log_operation(module, args, outputs)
        return outputs

    # 安全地获取模块名称
    module_name = module._module_name

    module.forward = wrapped_forward

def register_module_info(module: torch.nn.Module, prefix: str = ''):
    """为模块注册名称信息并包装forward方法"""
    # 设置当前模块的名称
    module._module_name = prefix if prefix else module.__class__.__name__
    
    # 包装forward方法
    wrap_forward(module)
    
    # 递归处理子模块
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        register_module_info(child, full_name)

# Llama模型实现
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
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=q.dtype))
        attn_weights = F.softmax(scores, dim=-1)
        
        # 应用注意力权重
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
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
        self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=config["layer_norm_eps"])
        self.post_attention_layernorm = nn.LayerNorm(self.hidden_size, eps=config["layer_norm_eps"])
    
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
        
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config["num_hidden_layers"])])
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
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
    
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
    # 创建模型
    model = create_test_model()
    
    # 注册模块信息
    register_module_info(model)
    
    # 测试输入
    batch_size = 1
    seq_len = 32
    input_ids = torch.randint(0, 32000, (batch_size, seq_len))
    
    # 清除日志记录
    torch_logger.clear_logs()
    
    # 前向传播
    with torch.no_grad():
        output = model(input_ids)
    
    # 保存日志
    torch_logger.dump_logs("torch_dump_logs")
    
    print(f"测试日志保存到 'torch_dump_logs' 目录")
    
    return output

def save_model_weights(model, output_path: str):
    """保存模型权重为safetensors格式"""
    # 获取权重
    state_dict = model.state_dict()
    
    # 使用WeightManager保存
    WeightManager.generate_shared_weights(
        get_llama_config(small=True), 
        output_path
    )
    
    print(f"模型权重保存到: {output_path}")

def load_model_with_weights(weights_path: str):
    """创建模型并加载权重"""
    # 创建模型
    model = create_test_model()
    
    # 加载权重
    model.load_weights(weights_path)
    
    return model

if __name__ == "__main__":
    # 测试权重保存和加载
    output_dir = "weights"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存权重文件
    weights_path = os.path.join(output_dir, "llama_shared.safetensors")
    
    # 1. 生成统一权重文件
    WeightManager.generate_shared_weights(get_llama_config(small=True), weights_path)
    
    # 2. 创建模型并加载权重
    model = create_test_model()
    model.load_weights(weights_path)
    
    # 3. 验证模型
    inputs = torch.randint(0, 32000, (1, 10))
    with torch.no_grad():
        outputs = model(inputs)
    
    print(f"模型输出形状: {outputs.shape}")
    print("PyTorch模型权重加载测试完成!") 