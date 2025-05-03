import functools
import numpy as np
from mindspore import Tensor, Parameter
from precision_compare.mock.mock import BaseOperatorLogger
from pathlib import Path
from typing import Any


class MindSporeOperatorLogger(BaseOperatorLogger):
    """MindSpore操作日志记录器"""

    def __init__(self):
        super().__init__(framework="mindspore")

    def _process_tensor_data(self, data):
        """处理MindSpore张量数据"""
        if isinstance(data, Tensor):
            return data.asnumpy()
        elif isinstance(data, (list, tuple)):
            return [self._process_tensor_data(x) for x in data]
        return data

    def _get_module_parameters(self, module) -> dict:
        """获取模块的所有参数 - MindSpore版本"""
        params = {}
        for name in module._params:
            param = getattr(module, name)
            if isinstance(param, Parameter):
                params[name] = param.asnumpy()
        return params

    def _save_data(self, path: Path, data: Any):
        """保存数据 - MindSpore版本"""
        np.save(str(path), data)

    def log_operation(self, module, inputs, outputs):
        """记录操作"""
        module_name = module._prefix
        self._create_module_entry(module_name, self._get_module_parameters(module))

        # 处理输入数据
        if isinstance(inputs, (list, tuple)) and len(inputs) > 0:
            processed_inputs = self._process_tensor_data(
                inputs[0] if len(inputs) == 1 else inputs
            )
            self.logs[module_name]["inputs"].append(processed_inputs)

        # 处理输出数据
        processed_outputs = self._process_tensor_data(outputs)
        self.logs[module_name]["outputs"].append(processed_outputs)

    def _save_module_data(self, path, data):
        """保存模块数据到文件系统 - MindSpore版本"""
        # 保存输入数据
        if data["inputs"]:
            for idx, input_data in enumerate(data["inputs"]):
                input_data_path = path / f"{self.get_next_gid()}_inputs_{idx}.npy"
                self._save_data(input_data_path, input_data)

        # 分别保存每个参数
        if data["parameters"]:
            for param_name, param_array in data["parameters"].items():
                param_path = path / f"{self.get_next_gid()}_parameters_{param_name}.npy"
                self._save_data(param_path, param_array)

        # 保存输出数据
        if data["outputs"]:
            for idx, output_data in enumerate(data["outputs"]):
                output_data_path = path / f"{self.get_next_gid()}_outputs_{idx}.npy"
                self._save_data(output_data_path, output_data)


def register_ms_module(module, ms_logger, prefix: str = ""):
    """为模块注册名称信息并包装construct方法 - MindSpore版本"""
    # 包装construct方法
    module._prefix = prefix
    wrap_construct(module, ms_logger)

    # 递归处理子模块 - MindSpore版本
    for name, child in module._cells.items():
        if child is not None:
            full_name = f"{prefix}.{name}" if prefix else name
            register_ms_module(child, ms_logger, full_name)


def wrap_construct(module, ms_logger):
    """包装模块的construct方法 - MindSpore版本"""
    original_construct = module.construct

    @functools.wraps(original_construct)
    def wrapped_construct(*args, **kwargs):
        outputs = original_construct(*args, **kwargs)
        ms_logger.log_operation(module, args, outputs)
        return outputs

    module.construct = wrapped_construct
