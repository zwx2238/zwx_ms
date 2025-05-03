from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from pathlib import Path

import numpy as np
import torch
import mindspore as ms
from mindspore import Tensor, Parameter


class InjectedError:
    """注入错误的描述类"""

    def __init__(
        self,
        module_path: str,
        error_type: str,
        description: str,
        framework: str,
        expected_impact: str,
        params: Dict[str, Any],
    ):
        self.module_path = module_path
        self.error_type = error_type
        self.description = description
        self.framework = framework
        self.expected_impact = expected_impact
        self.params = params


class BaseErrorInjector(ABC):
    """错误注入器基类"""

    def __init__(self, framework: str):
        self.framework = framework

    @abstractmethod
    def inject_weight_noise(
        self, model: Any, module_path: str, scale: float
    ) -> InjectedError:
        """注入权重噪声"""
        pass

    @abstractmethod
    def inject_dtype_cast(
        self, model: Any, module_path: str, dtype: str
    ) -> InjectedError:
        """注入数据类型转换"""
        pass

    @abstractmethod
    def inject_activation_quantization(
        self, model: Any, module_path: str, bits: int
    ) -> InjectedError:
        """注入激活量化"""
        pass

    def get_module_by_path(self, model: Any, module_path: str) -> Any:
        """通过路径获取模块"""
        module = model
        for part in module_path.split("."):
            module = getattr(module, part)
        return module


class TorchErrorInjector(BaseErrorInjector):
    """PyTorch错误注入器"""

    def __init__(self):
        super().__init__("torch")

    def inject_weight_noise(
        self, model: Any, module_path: str, scale: float = 0.01
    ) -> InjectedError:
        """为PyTorch模型注入权重噪声"""
        module = self.get_module_by_path(model, module_path)

        with torch.no_grad():
            for param_name, param in module.named_parameters(recurse=False):
                noise = torch.randn_like(param) * scale
                param.data += noise
                print(f"PyTorch: 在{module_path}.{param_name}中注入了{scale}比例的噪声")

        return InjectedError(
            module_path=module_path,
            error_type="weight_noise",
            description=f"在{module_path}中注入了{scale}比例的高斯噪声",
            framework=self.framework,
            expected_impact="权重扰动导致输出偏差",
            params={"scale": scale},
        )

    def inject_dtype_cast(
        self, model: Any, module_path: str, dtype: str = "float16"
    ) -> InjectedError:
        """为PyTorch模型注入数据类型转换"""
        module = self.get_module_by_path(model, module_path)
        torch_dtype = getattr(torch, dtype)

        def new_forward(*args, **kwargs):
            result = original_forward(*args, **kwargs)
            if isinstance(result, torch.Tensor):
                return result.to(torch_dtype).to(torch.float32)
            return result

        original_forward = module.forward
        module.forward = new_forward
        print(f"PyTorch: 在{module_path}中注入了到{dtype}的类型转换")

        return InjectedError(
            module_path=module_path,
            error_type="dtype_cast",
            description=f"在{module_path}中注入了到{dtype}的类型转换",
            framework=self.framework,
            expected_impact="数据类型转换导致精度损失",
            params={"dtype": dtype},
        )

    def inject_activation_quantization(
        self, model: Any, module_path: str, bits: int = 4
    ) -> InjectedError:
        """为PyTorch模型注入激活量化"""
        module = self.get_module_by_path(model, module_path)

        def quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
            max_val = torch.max(torch.abs(x))
            scale = (2 ** (bits - 1) - 1) / (max_val + 1e-10)
            return torch.round(x * scale) / scale

        original_forward = module.forward

        def new_forward(*args, **kwargs):
            result = original_forward(*args, **kwargs)
            if isinstance(result, torch.Tensor):
                return quantize(result, bits)
            return result

        module.forward = new_forward
        print(f"PyTorch: 在{module_path}中注入了{bits}比特的激活量化")

        return InjectedError(
            module_path=module_path,
            error_type="activation_quantization",
            description=f"在{module_path}中注入了{bits}比特的激活量化",
            framework=self.framework,
            expected_impact="激活值量化导致精度损失",
            params={"bits": bits},
        )


class MindSporeErrorInjector(BaseErrorInjector):
    """MindSpore错误注入器"""

    def __init__(self):
        super().__init__("mindspore")

    def inject_weight_noise(
        self, model: Any, module_path: str, scale: float = 0.01
    ) -> InjectedError:
        """为MindSpore模型注入权重噪声"""
        module = self.get_module_by_path(model, module_path)

        for param_name in module._params:
            param = getattr(module, param_name)
            if isinstance(param, ms.Parameter):
                param_data = param.data.asnumpy()
                noise = np.random.randn(*param_data.shape) * scale
                new_data = param_data + noise
                new_param = ms.Parameter(
                    ms.Tensor(new_data, param.data.dtype), name=param_name
                )
                module._params[param_name] = new_param
                setattr(module, param_name, new_param)
                print(
                    f"MindSpore: 在{module_path}.{param_name}中注入了{scale}比例的噪声"
                )

        return InjectedError(
            module_path=module_path,
            error_type="weight_noise",
            description=f"在{module_path}中注入了{scale}比例的高斯噪声",
            framework=self.framework,
            expected_impact="权重扰动导致输出偏差",
            params={"scale": scale},
        )

    def inject_dtype_cast(
        self, model: Any, module_path: str, dtype: str = "float16"
    ) -> InjectedError:
        """为MindSpore模型注入数据类型转换"""
        module = self.get_module_by_path(model, module_path)
        ms_dtype_map = {
            "float16": ms.float16,
            "float32": ms.float32,
            "int8": ms.int8,
            "int32": ms.int32,
        }
        ms_dtype = ms_dtype_map.get(dtype, ms.float16)

        original_construct = module.construct

        def new_construct(*args, **kwargs):
            result = original_construct(*args, **kwargs)
            if isinstance(result, ms.Tensor):
                return result.astype(ms_dtype).astype(ms.float32)
            return result

        module.construct = new_construct
        print(f"MindSpore: 在{module_path}中注入了到{dtype}的类型转换")

        return InjectedError(
            module_path=module_path,
            error_type="dtype_cast",
            description=f"在{module_path}中注入了到{dtype}的类型转换",
            framework=self.framework,
            expected_impact="数据类型转换导致精度损失",
            params={"dtype": dtype},
        )

    def inject_activation_quantization(
        self, model: Any, module_path: str, bits: int = 4
    ) -> InjectedError:
        """为MindSpore模型注入激活量化"""
        module = self.get_module_by_path(model, module_path)

        def quantize(x: ms.Tensor, bits: int) -> ms.Tensor:
            max_val, _ = ms.ops.max(ms.ops.abs(x))
            scale = (2 ** (bits - 1) - 1) / (max_val + 1e-10)
            return ms.ops.round(x * scale) / scale

        original_construct = module.construct

        def new_construct(*args, **kwargs):
            result = original_construct(*args, **kwargs)
            if isinstance(result, ms.Tensor):
                return quantize(result, bits)
            return result

        module.construct = new_construct
        print(f"MindSpore: 在{module_path}中注入了{bits}比特的激活量化")

        return InjectedError(
            module_path=module_path,
            error_type="activation_quantization",
            description=f"在{module_path}中注入了{bits}比特的激活量化",
            framework=self.framework,
            expected_impact="激活值量化导致精度损失",
            params={"bits": bits},
        )

    def inject_pynative_graph_switch(
        self, model: Any, module_path: str
    ) -> InjectedError:
        """为MindSpore模型注入PyNative/Graph模式切换错误"""
        module = self.get_module_by_path(model, module_path)
        original_construct = module.construct

        def new_construct(*args, **kwargs):
            original_mode = ms.get_context("mode")
            try:
                if original_mode == ms.PYNATIVE_MODE:
                    ms.set_context(mode=ms.GRAPH_MODE)
                    print(f"MindSpore: 在{module_path}中临时切换到GRAPH_MODE")
                else:
                    ms.set_context(mode=ms.PYNATIVE_MODE)
                    print(f"MindSpore: 在{module_path}中临时切换到PYNATIVE_MODE")
                return original_construct(*args, **kwargs)
            finally:
                ms.set_context(mode=original_mode)

        module.construct = new_construct

        return InjectedError(
            module_path=module_path,
            error_type="pynative_graph_switch",
            description=f"在{module_path}中注入了PyNative/Graph模式切换",
            framework=self.framework,
            expected_impact="执行模式切换导致的精度差异",
            params={},
        )

    def inject_tensor_layout(
        self, model: Any, module_path: str, layout: str = "NCHW"
    ) -> InjectedError:
        """为MindSpore模型注入张量布局转换错误"""
        module = self.get_module_by_path(model, module_path)
        original_construct = module.construct

        def transform_nchw(result: ms.Tensor) -> ms.Tensor:
            """NHWC -> NCHW -> NHWC 转换"""
            if len(result.shape) == 4:
                transposed = ms.ops.transpose(result, (0, 3, 1, 2))
                return ms.ops.transpose(transposed, (0, 2, 3, 1))
            return result

        def transform_nhwc(result: ms.Tensor) -> ms.Tensor:
            """NCHW -> NHWC -> NCHW 转换"""
            if len(result.shape) == 4:
                transposed = ms.ops.transpose(result, (0, 2, 3, 1))
                return ms.ops.transpose(transposed, (0, 3, 1, 2))
            return result

        def new_construct(*args, **kwargs):
            result = original_construct(*args, **kwargs)
            if isinstance(result, ms.Tensor) and len(result.shape) >= 4:
                print(f"MindSpore: 在{module_path}中进行{layout}布局转换")
                if layout == "NCHW":
                    result = transform_nchw(result)
                elif layout == "NHWC":
                    result = transform_nhwc(result)
            return result

        module.construct = new_construct

        return InjectedError(
            module_path=module_path,
            error_type="tensor_layout",
            description=f"在{module_path}中注入了{layout}布局转换",
            framework=self.framework,
            expected_impact="布局转换导致的精度损失",
            params={"layout": layout},
        )

    def inject_mixed_precision(
        self, model: Any, module_path: str, enable: bool = True
    ) -> InjectedError:
        """为MindSpore模型注入混合精度训练"""
        module = self.get_module_by_path(model, module_path)
        original_construct = module.construct

        def new_construct(*args, **kwargs):
            result = original_construct(*args, **kwargs)
            if isinstance(result, ms.Tensor) and enable:
                print(f"MindSpore: 在{module_path}中模拟混合精度训练")
                result = result.astype(ms.float16).astype(ms.float32)
            return result

        module.construct = new_construct

        return InjectedError(
            module_path=module_path,
            error_type="mixed_precision",
            description=f"在{module_path}中注入了混合精度模拟",
            framework=self.framework,
            expected_impact="混合精度计算导致的精度损失",
            params={"enable": enable},
        )


class ErrorInjectorFactory:
    """错误注入器工厂类"""

    @staticmethod
    def create_injector(framework: str) -> Optional[BaseErrorInjector]:
        """创建错误注入器实例"""
        if framework == "torch":
            return TorchErrorInjector()
        elif framework == "mindspore":
            return MindSporeErrorInjector()
        return None

    @staticmethod
    def create_error_injector(config: Dict, framework: str) -> Optional[callable]:
        """根据配置创建错误注入函数"""
        error_type = config.get("error_type", "none")
        module_path = config.get("module_path", "")

        if error_type == "none" or framework not in ["torch", "mindspore"]:
            return None

        injector = ErrorInjectorFactory.create_injector(framework)
        if not injector:
            return None

        if error_type == "weight_noise":
            scale = config.get("scale", 0.01)
            return lambda m: injector.inject_weight_noise(m, module_path, scale)
        elif error_type == "dtype_cast":
            dtype = config.get("dtype", "float16")
            return lambda m: injector.inject_dtype_cast(m, module_path, dtype)
        elif error_type == "activation_quantization":
            bits = config.get("bits", 4)
            return lambda m: injector.inject_activation_quantization(
                m, module_path, bits
            )
        elif error_type == "pynative_graph_switch" and framework == "mindspore":
            return lambda m: injector.inject_pynative_graph_switch(m, module_path)
        elif error_type == "tensor_layout" and framework == "mindspore":
            layout = config.get("layout", "NCHW")
            return lambda m: injector.inject_tensor_layout(m, module_path, layout)
        elif error_type == "mixed_precision" and framework == "mindspore":
            enable = config.get("enable", True)
            return lambda m: injector.inject_mixed_precision(m, module_path, enable)

        return None


class ErrorInjector:
    """为了保持向后兼容的错误注入器包装类"""

    @staticmethod
    def create_error_injector(config: Dict, framework: str) -> Optional[callable]:
        """
        根据配置创建错误注入器 - 向后兼容的接口

        Args:
            config: 错误注入配置
            framework: 框架名称 ('torch' 或 'mindspore')

        Returns:
            callable: 错误注入函数，如果不需要注入错误则返回None
        """
        return ErrorInjectorFactory.create_error_injector(config, framework)

    # 为了向后兼容，保留所有静态方法的引用
    inject_weight_noise_torch = staticmethod(
        lambda model, module_path, scale=0.01: TorchErrorInjector().inject_weight_noise(
            model, module_path, scale
        )
    )

    inject_weight_noise_mindspore = staticmethod(
        lambda model,
        module_path,
        scale=0.01: MindSporeErrorInjector().inject_weight_noise(
            model, module_path, scale
        )
    )

    inject_dtype_cast_torch = staticmethod(
        lambda model,
        module_path,
        dtype="float16": TorchErrorInjector().inject_dtype_cast(
            model, module_path, dtype
        )
    )

    inject_dtype_cast_mindspore = staticmethod(
        lambda model,
        module_path,
        dtype="float16": MindSporeErrorInjector().inject_dtype_cast(
            model, module_path, dtype
        )
    )

    inject_activation_quantization_torch = staticmethod(
        lambda model,
        module_path,
        bits=4: TorchErrorInjector().inject_activation_quantization(
            model, module_path, bits
        )
    )

    inject_activation_quantization_mindspore = staticmethod(
        lambda model,
        module_path,
        bits=4: MindSporeErrorInjector().inject_activation_quantization(
            model, module_path, bits
        )
    )

    inject_pynative_graph_switch_mindspore = staticmethod(
        lambda model,
        module_path: MindSporeErrorInjector().inject_pynative_graph_switch(
            model, module_path
        )
    )

    inject_tensor_layout_mindspore = staticmethod(
        lambda model,
        module_path,
        layout="NCHW": MindSporeErrorInjector().inject_tensor_layout(
            model, module_path, layout
        )
    )

    inject_mixed_precision_mindspore = staticmethod(
        lambda model,
        module_path,
        enable=True: MindSporeErrorInjector().inject_mixed_precision(
            model, module_path, enable
        )
    )
