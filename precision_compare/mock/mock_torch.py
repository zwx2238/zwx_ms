import functools
import torch
from pathlib import Path
from typing import Any

from precision_compare.mock.mock import BaseOperatorLogger


class TorchOperatorLogger(BaseOperatorLogger):
    """PyTorch操作日志记录器"""

    def __init__(self):
        super().__init__(framework="pytorch")

    def _process_tensor_data(self, data):
        """处理PyTorch张量数据"""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu()
        elif isinstance(data, (list, tuple)):
            return [self._process_tensor_data(x) for x in data]
        return data

    def _get_module_parameters(self, module: torch.nn.Module) -> dict:
        """获取模块的所有参数"""
        params = {}
        for name, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                params[name] = param.detach().cpu()
        return params

    def _save_data(self, path: Path, data: Any):
        """保存数据 - PyTorch版本"""
        torch.save(data, str(path))

    def log_operation(
        self, module: torch.nn.Module, inputs: tuple, outputs: torch.Tensor
    ):
        """记录操作"""
        module_name = module._prefix
        self._create_module_entry(module_name, self._get_module_parameters(module))

        # 处理输入数据
        if len(inputs) > 0:
            processed_inputs = self._process_tensor_data(
                inputs[0] if len(inputs) == 1 else inputs
            )
            self.logs[module_name]["inputs"].append(processed_inputs)

        # 处理输出数据
        processed_outputs = self._process_tensor_data(outputs)
        self.logs[module_name]["outputs"].append(processed_outputs)

    def _save_module_data(self, path, data):
        """保存模块数据到文件系统"""
        # 保存输入数据
        if data["inputs"]:
            for idx, input_data in enumerate(data["inputs"]):
                input_data_path = path / f"{self.get_next_gid()}_inputs_{idx}.pt"
                torch.save(input_data, str(input_data_path))

        # 分别保存每个参数
        if data["parameters"]:
            for param_name, param_tensor in data["parameters"].items():
                param_path = path / f"{self.get_next_gid()}_parameters_{param_name}.pt"
                torch.save(param_tensor, str(param_path))

        # 保存输出数据
        if data["outputs"]:
            for idx, output_data in enumerate(data["outputs"]):
                output_data_path = path / f"{self.get_next_gid()}_outputs_{idx}.pt"
                torch.save(output_data, str(output_data_path))


def register_torch_module(module: torch.nn.Module, torch_logger, prefix: str = ""):
    """为模块注册名称信息并包装forward方法"""
    # 设置当前模块的名称
    module._prefix = prefix
    # 包装forward方法
    wrap_forward(module, torch_logger)

    # 递归处理子模块
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        register_torch_module(child, torch_logger, full_name)


def wrap_forward(module: torch.nn.Module, torch_logger):
    """包装模块的forward方法"""
    original_forward = module.forward

    @functools.wraps(original_forward)
    def wrapped_forward(*args, **kwargs):
        outputs = original_forward(*args, **kwargs)
        torch_logger.log_operation(module, args, outputs)
        return outputs

    # 安全地获取模块名称
    module.forward = wrapped_forward
