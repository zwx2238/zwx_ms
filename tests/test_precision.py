import torch
import mindspore as ms
import numpy as np
import random
import os
from pathlib import Path
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import zwx_ms.mock.mock_ms
import zwx_ms.mock.mock_torch


# 自定义模块导入
from zwx_ms.analysis import analyze_runs
from zwx_ms.model.torch.llama import create_test_model as create_torch_model
import zwx_ms.model.torch.llama as torch_llama
from zwx_ms.model.mindspore.llama import create_test_model as create_ms_model
from zwx_ms.mock.mock_ms import register_module_info_ms as register_ms_module
from zwx_ms.mock.mock_torch import register_module_info_pt as register_torch_module
import zwx_ms.model.mindspore.llama as ms_llama

@dataclass
class InjectedError:
    """注入错误的信息"""
    module_path: str
    error_type: str
    description: str
    framework: str  # 'torch' 或 'mindspore'
    expected_impact: str
    params: Dict[str, Any] = None  # 存储错误注入的参数

class ErrorInjector:
    """错误注入器，支持PyTorch和MindSpore"""
    
    @staticmethod
    def inject_weight_noise_torch(model, module_path: str, scale: float = 0.01) -> InjectedError:
        """为PyTorch模型注入权重噪声"""
        module = model
        for part in module_path.split('.'):
            module = getattr(module, part)
        
        with torch.no_grad():
            for param_name, param in module.named_parameters(recurse=False):
                noise = torch.randn_like(param) * scale
                param.data += noise
                print(f"PyTorch: 在{module_path}.{param_name}中注入了{scale}比例的噪声")
        
        return InjectedError(
            module_path=module_path,
            error_type="weight_noise",
            description=f"在{module_path}中注入了{scale}比例的高斯噪声",
            framework="torch",
            expected_impact="权重扰动导致输出偏差",
            params={"scale": scale}
        )
    
    @staticmethod
    def inject_weight_noise_mindspore(model, module_path: str, scale: float = 0.01) -> InjectedError:
        """为MindSpore模型注入权重噪声"""
        module = model
        path_parts = module_path.split('.')
        
        # 遍历路径获取模块
        for part in path_parts:
            module = getattr(module, part)
        
        # 扰动参数
        for param_name in module._params:
            param = getattr(module, param_name)
            if isinstance(param, ms.Parameter):
                # 提取参数值并添加噪声
                param_data = param.data.asnumpy()
                noise = np.random.randn(*param_data.shape) * scale
                new_data = param_data + noise
                
                # 更新参数
                new_param = ms.Parameter(ms.Tensor(new_data, param.data.dtype), name=param_name)
                module._params[param_name] = new_param
                setattr(module, param_name, new_param)
                print(f"MindSpore: 在{module_path}.{param_name}中注入了{scale}比例的噪声")
        
        return InjectedError(
            module_path=module_path,
            error_type="weight_noise",
            description=f"在{module_path}中注入了{scale}比例的高斯噪声",
            framework="mindspore",
            expected_impact="权重扰动导致输出偏差",
            params={"scale": scale}
        )
    
    @staticmethod
    def inject_dtype_cast_torch(model, module_path: str, dtype: str = "float16") -> InjectedError:
        """为PyTorch模型注入数据类型转换"""
        module = model
        for part in module_path.split('.'):
            module = getattr(module, part)
        
        # 获取对应的torch dtype
        torch_dtype = getattr(torch, dtype)
        
        # 保存原始forward方法
        original_forward = module.forward
        
        # 创建新的forward方法
        def new_forward(*args, **kwargs):
            result = original_forward(*args, **kwargs)
            if isinstance(result, torch.Tensor):
                return result.to(torch_dtype).to(torch.float32)
            return result
        
        # 替换forward方法
        module.forward = new_forward
        print(f"PyTorch: 在{module_path}中注入了到{dtype}的类型转换")
        
        return InjectedError(
            module_path=module_path,
            error_type="dtype_cast",
            description=f"在{module_path}中注入了到{dtype}的类型转换",
            framework="torch",
            expected_impact="数据类型转换导致精度损失",
            params={"dtype": dtype}
        )
    
    @staticmethod
    def inject_dtype_cast_mindspore(model, module_path: str, dtype: str = "float16") -> InjectedError:
        """为MindSpore模型注入数据类型转换"""
        module = model
        for part in module_path.split('.'):
            module = getattr(module, part)
        
        # 获取对应的mindspore dtype
        ms_dtype_map = {
            "float16": ms.float16,
            "float32": ms.float32,
            "int8": ms.int8,
            "int32": ms.int32
        }
        ms_dtype = ms_dtype_map.get(dtype, ms.float16)
        
        # 保存原始construct方法
        original_construct = module.construct
        
        # 创建新的construct方法
        def new_construct(*args, **kwargs):
            result = original_construct(*args, **kwargs)
            if isinstance(result, ms.Tensor):
                return result.astype(ms_dtype).astype(ms.float32)
            return result
        
        # 替换construct方法
        module.construct = new_construct
        print(f"MindSpore: 在{module_path}中注入了到{dtype}的类型转换")
        
        return InjectedError(
            module_path=module_path,
            error_type="dtype_cast",
            description=f"在{module_path}中注入了到{dtype}的类型转换",
            framework="mindspore",
            expected_impact="数据类型转换导致精度损失",
            params={"dtype": dtype}
        )
    
    @staticmethod
    def inject_activation_quantization_torch(model, module_path: str, bits: int = 4) -> InjectedError:
        """为PyTorch模型注入激活量化"""
        module = model
        for part in module_path.split('.'):
            module = getattr(module, part)
        
        def quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
            max_val = torch.max(torch.abs(x))
            scale = (2 ** (bits - 1) - 1) / (max_val + 1e-10)
            return torch.round(x * scale) / scale
        
        # 保存原始forward方法
        original_forward = module.forward
        
        # 创建新的forward方法
        def new_forward(*args, **kwargs):
            result = original_forward(*args, **kwargs)
            if isinstance(result, torch.Tensor):
                return quantize(result, bits)
            return result
        
        # 替换forward方法
        module.forward = new_forward
        print(f"PyTorch: 在{module_path}中注入了{bits}比特的激活量化")
        
        return InjectedError(
            module_path=module_path,
            error_type="activation_quantization",
            description=f"在{module_path}中注入了{bits}比特的激活量化",
            framework="torch",
            expected_impact="激活值量化导致精度损失",
            params={"bits": bits}
        )
    
    @staticmethod
    def inject_activation_quantization_mindspore(model, module_path: str, bits: int = 4) -> InjectedError:
        """为MindSpore模型注入激活量化"""
        module = model
        for part in module_path.split('.'):
            module = getattr(module, part)
        
        def quantize(x: ms.Tensor, bits: int) -> ms.Tensor:
            max_val = ms.ops.maximum(ms.ops.abs(x))
            scale = (2 ** (bits - 1) - 1) / (max_val + 1e-10)
            return ms.ops.round(x * scale) / scale
        
        # 保存原始construct方法
        original_construct = module.construct
        
        # 创建新的construct方法
        def new_construct(*args, **kwargs):
            result = original_construct(*args, **kwargs)
            if isinstance(result, ms.Tensor):
                return quantize(result, bits)
            return result
        
        # 替换construct方法
        module.construct = new_construct
        print(f"MindSpore: 在{module_path}中注入了{bits}比特的激活量化")
        
        return InjectedError(
            module_path=module_path,
            error_type="activation_quantization",
            description=f"在{module_path}中注入了{bits}比特的激活量化",
            framework="mindspore",
            expected_impact="激活值量化导致精度损失",
            params={"bits": bits}
        )

    @staticmethod
    def inject_pynative_graph_switch_mindspore(model, module_path: str) -> InjectedError:
        """为MindSpore模型注入PyNative/Graph模式切换错误
        这会模拟在模型部分组件上使用不同的执行模式可能导致的精度问题
        """
        module = model
        for part in module_path.split('.'):
            module = getattr(module, part)
        
        # 保存原始construct方法
        original_construct = module.construct
        
        # 创建新的construct方法，在内部切换执行模式
        def new_construct(*args, **kwargs):
            # 临时切换到图模式
            original_mode = ms.get_context("mode")
            try:
                # 如果当前是PyNative模式，临时切换到图模式，反之亦然
                if original_mode == ms.PYNATIVE_MODE:
                    ms.set_context(mode=ms.GRAPH_MODE)
                    print(f"MindSpore: 在{module_path}中临时切换到GRAPH_MODE")
                else:
                    ms.set_context(mode=ms.PYNATIVE_MODE)
                    print(f"MindSpore: 在{module_path}中临时切换到PYNATIVE_MODE")
                
                # 执行原始函数
                result = original_construct(*args, **kwargs)
                return result
            finally:
                # 恢复原始执行模式
                ms.set_context(mode=original_mode)
        
        # 替换construct方法
        module.construct = new_construct
        
        return InjectedError(
            module_path=module_path,
            error_type="pynative_graph_switch",
            description=f"在{module_path}中注入了PyNative/Graph模式切换",
            framework="mindspore",
            expected_impact="执行模式切换导致的精度差异",
            params={}
        )

    @staticmethod
    def inject_tensor_layout_mindspore(model, module_path: str, layout: str = "NCHW") -> InjectedError:
        """为MindSpore模型注入张量布局转换错误"""
        module = model
        for part in module_path.split('.'):
            module = getattr(module, part)
        
        # 保存原始construct方法
        original_construct = module.construct
        
        # 创建新的construct方法，在内部进行布局转换
        def new_construct(*args, **kwargs):
            # 执行原始函数
            result = original_construct(*args, **kwargs)
            
            if isinstance(result, ms.Tensor) and len(result.shape) >= 4:
                print(f"MindSpore: 在{module_path}中进行{layout}布局转换")
                # 模拟布局转换带来的精度影响
                if layout == "NCHW":
                    # NHWC -> NCHW -> NHWC 转换会带来精度损失
                    shape = result.shape
                    if len(shape) == 4:
                        # 假设输入是NHWC (batch, height, width, channel)
                        # 转为NCHW (batch, channel, height, width)
                        transposed = ms.ops.transpose(result, (0, 3, 1, 2))
                        # 再转回NHWC
                        result = ms.ops.transpose(transposed, (0, 2, 3, 1))
                elif layout == "NHWC":
                    # NCHW -> NHWC -> NCHW 转换
                    shape = result.shape
                    if len(shape) == 4:
                        # 假设输入是NCHW (batch, channel, height, width)
                        # 转为NHWC (batch, height, width, channel)
                        transposed = ms.ops.transpose(result, (0, 2, 3, 1))
                        # 再转回NCHW
                        result = ms.ops.transpose(transposed, (0, 3, 1, 2))
            
            return result
        
        # 替换construct方法
        module.construct = new_construct
        
        return InjectedError(
            module_path=module_path,
            error_type="tensor_layout",
            description=f"在{module_path}中注入了{layout}布局转换",
            framework="mindspore",
            expected_impact="布局转换导致的精度损失",
            params={"layout": layout}
        )

    @staticmethod
    def inject_mixed_precision_mindspore(model, module_path: str, enable: bool = True) -> InjectedError:
        """为MindSpore模型注入混合精度训练"""
        module = model
        for part in module_path.split('.'):
            module = getattr(module, part)
        
        # 保存原始construct方法
        original_construct = module.construct
        
        # 创建新的construct方法，在内部使用混合精度
        def new_construct(*args, **kwargs):
            # 执行原始函数
            result = original_construct(*args, **kwargs)
            
            if isinstance(result, ms.Tensor):
                print(f"MindSpore: 在{module_path}中模拟混合精度训练")
                # 模拟混合精度：FP32->FP16->FP32转换
                if enable:
                    result = result.astype(ms.float16).astype(ms.float32)
            
            return result
        
        # 替换construct方法
        module.construct = new_construct
        
        return InjectedError(
            module_path=module_path,
            error_type="mixed_precision",
            description=f"在{module_path}中注入了混合精度模拟",
            framework="mindspore",
            expected_impact="混合精度计算导致的精度损失",
            params={"enable": enable}
        )

def create_test_case(case_name: str, config: Dict) -> str:
    """创建测试用例目录"""
    # 获取错误类型
    error_type = config.get("error_type", "none")
    module_path = config.get("module_path", "")
    
    # 使用日期后缀确保唯一性
    date_suffix = datetime.now().strftime("%Y%m%d")
    
    # 创建目录名
    dir_name = f"{case_name}_{error_type}_{date_suffix}"
    base_dir = Path("test_cases") / dir_name
    
    # 创建测试目录
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建分析结果目录
    analysis_dir = base_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    # 保存错误配置
    with open(base_dir / "test_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    return str(base_dir)

def set_random_seed(seed: int = 42):
    """设置所有随机种子以确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    if torch:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    if ms:
        ms.set_seed(seed)

def run_pytorch_model(save_dir: str, error_injector: Optional = None, input_data: np.ndarray = None) -> np.ndarray:
    """运行PyTorch模型并保存结果"""
    torch_save_dir = os.path.join(save_dir, "torch")
    os.makedirs(torch_save_dir, exist_ok=True)
    weights_dir = os.path.join(save_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    # 创建模型
    model = create_torch_model()
    
    # 生成共享权重文件（如果不存在）
    weights_path = os.path.join(weights_dir, "shared_weights.safetensors")
    if not os.path.exists(weights_path):
        from zwx_ms.utils.weight_utils import WeightManager
        config = torch_llama.get_llama_config(small=True)
        WeightManager.generate_shared_weights(config, weights_path, seed=42)
    
    # 加载共享权重
    try:
        model.load_weights(weights_path)
        print("权重加载成功")
    except Exception as e:
        print(f"权重加载失败: {str(e)}")
    
    # 注入错误（如果有）
    if error_injector:
        print("注入错误到PyTorch模型...")
        error_info = error_injector(model)
        # 保存错误信息
        error_info_path = Path(torch_save_dir) / "injected_error.json"
        with open(error_info_path, "w", encoding="utf-8") as f:
            json.dump(error_info.__dict__, f, indent=2, ensure_ascii=False)
    
    # 注册模块信息
    register_torch_module(model)
    
    # 运行模型
    input_ids = torch.tensor(input_data, dtype=torch.long)
    
    # 清除已有日志
    zwx_ms.mock.mock_torch.torch_logger.clear_logs()
    
    # 前向传播
    with torch.no_grad():
        outputs = model(input_ids)
    
    # 保存日志
    zwx_ms.mock.mock_torch.torch_logger.dump_logs(torch_save_dir)
    
    # 保存输入和输出
    np.save(os.path.join(save_dir, "torch_input.npy"), input_ids.numpy())
    np.save(os.path.join(save_dir, "torch_output.npy"), outputs.detach().numpy())
    
    return outputs.detach().numpy()

def run_mindspore_model(save_dir: str, error_injector: Optional = None, input_data: np.ndarray = None) -> np.ndarray:
    """运行MindSpore模型并保存结果"""
    # 设置MindSpore上下文
    ms.set_context(mode=ms.PYNATIVE_MODE)
    
    ms_save_dir = os.path.join(save_dir, "mindspore")
    os.makedirs(ms_save_dir, exist_ok=True)
    weights_dir = os.path.join(save_dir, "weights")
    
    # 创建模型
    model = create_ms_model()
    
    # 加载共享权重
    weights_path = os.path.join(weights_dir, "shared_weights.safetensors")
    try:
        model.load_weights(weights_path)
        print("权重加载成功")
    except Exception as e:
        print(f"权重加载失败: {str(e)}")
    
    # 注入错误（如果有）
    if error_injector:
        print("注入错误到MindSpore模型...")
        error_info = error_injector(model)
        # 保存错误信息
        error_info_path = Path(ms_save_dir) / "injected_error.json"
        with open(error_info_path, "w", encoding="utf-8") as f:
            json.dump(error_info.__dict__, f, indent=2, ensure_ascii=False)
    
    # 注册模块信息
    register_ms_module(model)

    input_ids = ms.Tensor(input_data, ms.int32)
    
    # 清除已有日志
    zwx_ms.mock.mock_ms.ms_logger.clear_logs()
    
    # 前向传播
    outputs = model(input_ids)
    
    # 保存日志
    zwx_ms.mock.mock_ms.ms_logger.dump_logs(ms_save_dir)
    
    # 保存输入和输出
    np.save(os.path.join(save_dir, "mindspore_input.npy"), input_ids.asnumpy())
    np.save(os.path.join(save_dir, "mindspore_output.npy"), outputs.asnumpy())
    
    return outputs.asnumpy()

def compare_outputs(torch_output: np.ndarray, ms_output: np.ndarray) -> Dict:
    """比较两个框架的输出差异"""
    # 形状和数据类型信息
    torch_shape = torch_output.shape
    ms_shape = ms_output.shape
    torch_dtype = str(torch_output.dtype)
    ms_dtype = str(ms_output.dtype)
    
    # 检查形状是否匹配
    if torch_shape != ms_shape:
        return {
            "error": f"形状不匹配: PyTorch {torch_shape} vs MindSpore {ms_shape}",
            "torch_shape": torch_shape,
            "ms_shape": ms_shape,
            "torch_dtype": torch_dtype,
            "ms_dtype": ms_dtype
        }
    
    # 计算差异
    diff = np.abs(torch_output - ms_output)
    
    # 1. 最大绝对误差
    max_diff = float(np.max(diff))
    
    # 2. 找到最大差异的位置
    max_idx = np.unravel_index(np.argmax(diff), diff.shape)
    torch_value = float(torch_output[max_idx])
    ms_value = float(ms_output[max_idx])
    
    # 3. 最大差异位置的相对误差
    rel_err_at_max = float(diff[max_idx] / (max(abs(torch_value), abs(ms_value)) + 1e-10))
    
    # 4. 余弦相似度
    # 将数组展平以计算余弦相似度
    torch_flat = torch_output.flatten()
    ms_flat = ms_output.flatten()
    
    # 计算余弦相似度 = (a·b) / (||a|| * ||b||)
    dot_product = np.sum(torch_flat * ms_flat)
    torch_norm = np.sqrt(np.sum(torch_flat ** 2))
    ms_norm = np.sqrt(np.sum(ms_flat ** 2))
    
    cosine_similarity = float(dot_product / (torch_norm * ms_norm + 1e-10))
    
    # 返回关键指标
    return {
        # 形状和数据类型信息
        "torch_shape": torch_shape,
        "ms_shape": ms_shape,
        "torch_dtype": torch_dtype,
        "ms_dtype": ms_dtype,
        
        # 关键指标
        "max_abs_diff": max_diff,
        "cosine_similarity": cosine_similarity,
        "rel_err_at_max_diff": rel_err_at_max,
        
        # 最大差异的位置信息
        "max_diff_location": [int(i) for i in max_idx],
        "torch_value_at_max": torch_value,
        "ms_value_at_max": ms_value
    }

def run_test_case(test_dir: str, config: Dict) -> None:
    """运行单个测试用例"""
    test_dir_path = Path(test_dir)
    print(f"\n运行测试用例: {test_dir_path.name}")
    
    # 设置随机种子
    set_random_seed(42)
    
    # 创建测试目录
    os.makedirs(test_dir, exist_ok=True)
    
    # 保存测试配置
    with open(os.path.join(test_dir, "test_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # 获取错误注入配置
    error_type = config.get("error_type", "none")
    module_path = config.get("module_path", "")
    framework = config.get("framework", "both")  # 选择注入错误的框架: torch, mindspore, both
    
    # 首先生成固定的输入数据，确保两个框架使用相同的输入
    print("生成统一输入数据...")
    batch_size = 1
    seq_len = 10
    # 使用固定种子生成随机输入
    np.random.seed(42)
    input_data = np.random.randint(0, 32000, (batch_size, seq_len))
    
    # 分别运行PyTorch和MindSpore模型
    print("运行PyTorch模型...")
    torch_error_injector = None
    if error_type != "none" and framework in ["torch", "both"]:
        # 选择合适的错误注入方法
        if error_type == "weight_noise":
            scale = config.get("scale", 0.01)
            torch_error_injector = lambda m: ErrorInjector.inject_weight_noise_torch(m, module_path, scale)
        elif error_type == "dtype_cast":
            dtype = config.get("dtype", "float16")
            torch_error_injector = lambda m: ErrorInjector.inject_dtype_cast_torch(m, module_path, dtype)
        elif error_type == "activation_quantization":
            bits = config.get("bits", 4)
            torch_error_injector = lambda m: ErrorInjector.inject_activation_quantization_torch(m, module_path, bits)
    
    # 传递共享输入数据路径
    torch_output = run_pytorch_model(test_dir, torch_error_injector, input_data=input_data)
    
    print("\n运行MindSpore模型...")
    ms_error_injector = None
    if error_type != "none" and framework in ["mindspore", "both"]:
        # 选择合适的错误注入方法
        if error_type == "weight_noise":
            scale = config.get("scale", 0.01)
            ms_error_injector = lambda m: ErrorInjector.inject_weight_noise_mindspore(m, module_path, scale)
        elif error_type == "dtype_cast":
            dtype = config.get("dtype", "float16")
            ms_error_injector = lambda m: ErrorInjector.inject_dtype_cast_mindspore(m, module_path, dtype)
        elif error_type == "activation_quantization":
            bits = config.get("bits", 4)
            ms_error_injector = lambda m: ErrorInjector.inject_activation_quantization_mindspore(m, module_path, bits)
        elif error_type == "pynative_graph_switch":
            ms_error_injector = lambda m: ErrorInjector.inject_pynative_graph_switch_mindspore(m, module_path)
        elif error_type == "tensor_layout":
            layout = config.get("layout", "NCHW")
            ms_error_injector = lambda m: ErrorInjector.inject_tensor_layout_mindspore(m, module_path, layout)
        elif error_type == "mixed_precision":
            enable = config.get("enable", True)
            ms_error_injector = lambda m: ErrorInjector.inject_mixed_precision_mindspore(m, module_path, enable)
    
    # 传递共享输入数据路径
    ms_output = run_mindspore_model(test_dir, ms_error_injector, input_data=input_data)
    
    # 比较两个模型的输出
    print("\n比较两个框架的输出差异...")
    comparison = compare_outputs(torch_output, ms_output)
    
    # 保存比较结果
    with open(os.path.join(test_dir, "comparison_results.json"), "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    # 打印主要差异指标
    print("\n===== 精度对比关键指标 =====")
    print(f"输入形状: {comparison.get('torch_shape')}")
    print(f"PyTorch dtype: {comparison.get('torch_dtype')}, MindSpore dtype: {comparison.get('ms_dtype')}")
    print(f"最大绝对误差: {comparison.get('max_abs_diff'):.6e}")
    print(f"余弦相似度: {comparison.get('cosine_similarity'):.6f}")
    print(f"最大误差位置 {comparison.get('max_diff_location')} 的相对误差: {comparison.get('rel_err_at_max_diff'):.6%}")
    print(f"最大误差位置的值 - PyTorch: {comparison.get('torch_value_at_max'):.6f}, MindSpore: {comparison.get('ms_value_at_max'):.6f}")
    print("===========================")
    
    # 分析模型内部差异
    print("\n开始详细分析内部差异...")
    try:
        result = analyze_runs(
            os.path.join(test_dir, "torch"), 
            os.path.join(test_dir, "mindspore"), 
            os.path.join(test_dir, "analysis")
        )
        
        # 打印分析结果摘要
        print("\n===== 内部层分析摘要 =====")
        if 'module_comparisons' in result:
            modules = result['module_comparisons']
            if modules:
                max_diff_module = max(modules.items(), key=lambda x: x[1].get('max_abs_diff', 0))
                print(f"最大差异模块: {max_diff_module[0]}, 绝对误差: {max_diff_module[1].get('max_abs_diff', 0):.6e}")
                
                # 输出余弦相似度最小的模块
                min_cosine_module = min(modules.items(), key=lambda x: x[1].get('cosine_similarity', 1.0))
                print(f"余弦相似度最低模块: {min_cosine_module[0]}, 相似度: {min_cosine_module[1].get('cosine_similarity', 0):.6f}")
            else:
                print("未找到模块比较结果")
        else:
            print("分析结果未包含模块比较信息")
        print("=========================")
        
        print(f"详细分析结果已保存到 {os.path.join(test_dir, 'analysis')}")
    except Exception as e:
        print(f"分析过程出错: {str(e)}")
        import traceback
        traceback.print_exc()

def run_parallel_tests(test_configs: List[Dict]) -> None:
    """并行运行多个测试用例"""
    print(f"\n开始执行 {len(test_configs)} 个测试用例...")
    
    # 创建测试用例目录
    test_cases = []
    for idx, config in enumerate(test_configs):
        case_name = f"case{idx+1}"
        test_dir = create_test_case(case_name, config)
        test_cases.append((test_dir, config))
        print(f"创建测试用例: {Path(test_dir).name}")
    
    # 使用进程池并行执行测试
    with ProcessPoolExecutor(max_workers=min(len(test_cases), os.cpu_count() or 1)) as executor:
        # 提交所有测试任务
        futures = []
        for test_dir, config in test_cases:
            future = executor.submit(run_test_case, test_dir, config)
            futures.append(future)
        
        # 等待所有任务完成
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"测试执行错误: {str(e)}")
    
    print("\n所有测试完成!")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试PyTorch和MindSpore模型精度差异")
    parser.add_argument("--parallel", action="store_true", help="并行执行多个测试")
    parser.add_argument("--error-type", type=str, default="weight_noise",
                       choices=["none", "weight_noise", "dtype_cast", "activation_quantization", 
                                "pynative_graph_switch", "tensor_layout", "mixed_precision"],
                       help="错误注入类型")
    parser.add_argument("--module-path", type=str, default="model.layers.1.input_layernorm",
                       help="注入错误的模块路径")
    parser.add_argument("--framework", type=str, default="both",
                       choices=["torch", "mindspore", "both"],
                       help="要注入错误的框架")
    parser.add_argument("--scale", type=float, default=0.01,
                       help="权重噪声的比例")
    parser.add_argument("--dtype", type=str, default="float16",
                       help="类型转换的目标类型")
    parser.add_argument("--bits", type=int, default=4,
                       help="激活量化的位数")
    parser.add_argument("--layout", type=str, default="NCHW",
                       choices=["NCHW", "NHWC"],
                       help="张量布局格式(用于tensor_layout错误类型)")
    parser.add_argument("--enable", action="store_true",
                       help="启用混合精度(用于mixed_precision错误类型)")
    
    args = parser.parse_args()
    
    # 构建测试配置
    test_config = {
        "error_type": args.error_type,
        "module_path": args.module_path,
        "framework": args.framework,
        "scale": args.scale,
        "dtype": args.dtype,
        "bits": args.bits,
        "layout": args.layout,
        "enable": args.enable
    }
    
    if args.parallel:
        # 多种测试配置
        test_configs = [
            # 基准比较
            {"error_type": "none", "framework": "both"},
            
            # MindSpore特有的错误类型
            {"error_type": "pynative_graph_switch", "module_path": "model.layers.0", "framework": "mindspore"},
            {"error_type": "tensor_layout", "module_path": "model.layers.0", "framework": "mindspore", "layout": "NCHW"},
            {"error_type": "tensor_layout", "module_path": "model.layers.0", "framework": "mindspore", "layout": "NHWC"},
            {"error_type": "mixed_precision", "module_path": "model", "framework": "mindspore", "enable": True},
            
            # 权重噪声测试
            {"error_type": "weight_noise", "module_path": "model.lm_head", "framework": "mindspore", "scale": 0.01},
            {"error_type": "weight_noise", "module_path": "model.lm_head", "framework": "torch", "scale": 0.01},
            
            # 类型转换测试
            {"error_type": "dtype_cast", "module_path": "model", "framework": "mindspore", "dtype": "float16"},
            
            # 激活量化测试
            {"error_type": "activation_quantization", "module_path": "model.layers.0", "framework": "mindspore", "bits": 4}
        ]
        run_parallel_tests(test_configs)
    else:
        # 运行单个测试
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_dir = f"test_case_{args.error_type}_{timestamp}"
        run_test_case(test_dir, test_config)

if __name__ == "__main__":
    main() 