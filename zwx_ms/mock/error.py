from dataclasses import dataclass
from typing import Dict, Any

import mindspore as ms
import numpy as np
import torch


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
