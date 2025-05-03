import json
from typing import Dict, Any, List, Tuple
from pathlib import Path


class BaseOperatorLogger:
    """基础操作日志记录器，为PyTorch和MindSpore提供通用功能"""

    def __init__(self):
        self.logs: Dict[str, Dict[str, Any]] = {}
        self.gid = 0
        self.exec_counter = 0  # 执行计数器

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

            # 创建每一层的目录
            current_path = output_path
            for part in module_parts:
                current_path = current_path / part
                if not current_path.exists():
                    current_path.mkdir(parents=False, exist_ok=True)

            # 具体的数据保存方法由子类实现
            self._save_module_data(current_path, data)

    def _save_module_data(self, path: Path, data: Dict[str, Any]):
        """保存模块数据，由子类实现具体存储方式"""
        raise NotImplementedError("子类必须实现此方法")
