import json
from typing import Dict, Any, List, Tuple
from pathlib import Path


class BaseOperatorLogger:
    """基础操作日志记录器，为PyTorch和MindSpore提供通用功能"""
    
    def __init__(self):
        self.logs: Dict[str, Dict[str, Any]] = {}
        self.execution_order: List[Tuple[str, str]] = []  # 记录执行顺序：(模块名, 操作类型)
        self.exec_counter = 0  # 执行计数器
    
    def clear_logs(self):
        """清除所有日志记录"""
        self.logs = {}
        self.execution_order = []
        self.exec_counter = 0
    
    def _create_module_entry(self, module_name: str, parameters: Dict[str, Any]):
        """创建模块条目"""
        if module_name not in self.logs:
            self.logs[module_name] = {
                'inputs': [],
                'outputs': [],
                'parameters': parameters,
                'execution_index': {}  # 记录每种操作的执行索引
            }
    
    def _log_execution_index(self, module_name: str, operation_type: str):
        """记录执行索引"""
        exec_idx = self.exec_counter
        self.execution_order.append((module_name, operation_type))
        self.logs[module_name]['execution_index'][operation_type] = exec_idx
        self.exec_counter += 1

    def dump_logs(self, output_dir: str = 'operator_logs'):
        """保存日志到文件系统"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 保存执行顺序信息
        exec_order_path = output_path / "execution_order.json"
        with open(exec_order_path, 'w', encoding='utf-8') as f:
            json.dump({
                'execution_order': self.execution_order,
                'module_execution_index': {
                    module_name: data.get('execution_index', {})
                    for module_name, data in self.logs.items()
                }
            }, f, indent=2, ensure_ascii=False)
        
        for module_name, data in self.logs.items():
            # 按照模块的层级结构创建目录
            module_parts = module_name.split('.')
            
            # 创建每一层的目录
            current_path = output_path
            for part in module_parts:
                current_path = current_path / part
                if not current_path.exists():
                    current_path.mkdir(parents=False, exist_ok=True)
            
            # 保存执行索引信息
            exec_info_path = current_path / "execution_index.json"
            with open(exec_info_path, 'w', encoding='utf-8') as f:
                json.dump(data.get('execution_index', {}), f, indent=2)
            
            # 具体的数据保存方法由子类实现
            self._save_module_data(current_path, data)
    
    def _save_module_data(self, path: Path, data: Dict[str, Any]):
        """保存模块数据，由子类实现具体存储方式"""
        raise NotImplementedError("子类必须实现此方法")


