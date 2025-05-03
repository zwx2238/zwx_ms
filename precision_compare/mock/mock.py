import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Literal
from pathlib import Path
import numpy as np


class BaseOperatorLogger(ABC):
    """基础操作日志记录器，为PyTorch和MindSpore提供通用功能"""

    def __init__(self, framework: Literal["pytorch", "mindspore"]):
        self.logs: Dict[str, Dict[str, Any]] = {}
        self.gid = 0
        self.exec_counter = 0  # 执行计数器
        self.framework = framework
        self.file_extension = ".pt" if framework == "pytorch" else ".npy"

    def clear_logs(self):
        """清除所有日志记录"""
        self.logs = {}
        self.exec_counter = 0

    def get_next_gid(self):
        """获取下一个全局ID"""
        self.gid += 1
        return self.gid

    def _create_module_entry(self, module_name: str, parameters: Dict[str, Any]):
        """创建模块条目"""
        if module_name not in self.logs:
            self.logs[module_name] = {
                "inputs": [],
                "outputs": [],
                "parameters": parameters,
                "execution_index": {},  # 记录每种操作的执行索引
            }

    def dump_logs(self, output_dir: str = "operator_logs"):
        """保存日志到文件系统"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        for module_name, data in self.logs.items():
            # 按照模块的层级结构创建目录
            module_parts = module_name.split(".")
            current_path = output_path
            for part in module_parts:
                current_path = current_path / part
                if not current_path.exists():
                    current_path.mkdir(parents=False, exist_ok=True)

            self._save_module_data(current_path, data)

    def log_operation(self, module, inputs, outputs):
        """通用的操作记录逻辑"""
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

    def _save_module_data(self, path: Path, data: Dict[str, Any]):
        """通用的模块数据保存逻辑"""
        # 保存输入数据
        if data["inputs"]:
            for idx, input_data in enumerate(data["inputs"]):
                input_data_path = (
                    path / f"{self.get_next_gid()}_inputs_{idx}{self.file_extension}"
                )
                self._save_data(input_data_path, input_data)

        # 保存参数
        if data["parameters"]:
            for param_name, param_data in data["parameters"].items():
                param_path = (
                    path
                    / f"{self.get_next_gid()}_parameters_{param_name}{self.file_extension}"
                )
                self._save_data(param_path, param_data)

        # 保存输出数据
        if data["outputs"]:
            for idx, output_data in enumerate(data["outputs"]):
                output_data_path = (
                    path / f"{self.get_next_gid()}_outputs_{idx}{self.file_extension}"
                )
                self._save_data(output_data_path, output_data)

    @abstractmethod
    def _process_tensor_data(self, data):
        """处理张量数据的抽象方法"""
        pass

    @abstractmethod
    def _get_module_parameters(self, module) -> dict:
        """获取模块参数的抽象方法"""
        pass

    @abstractmethod
    def _save_data(self, path: Path, data: Any):
        """保存数据的抽象方法"""
        pass
