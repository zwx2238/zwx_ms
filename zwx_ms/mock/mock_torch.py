import functools

import torch

from zwx_ms.mock.mock import BaseOperatorLogger


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


def register_module_info_pt(module: torch.nn.Module, prefix: str = ''):
    """为模块注册名称信息并包装forward方法"""
    # 设置当前模块的名称
    module._module_name = prefix if prefix else module.__class__.__name__

    # 包装forward方法
    wrap_forward(module)

    # 递归处理子模块
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        register_module_info_pt(child, full_name)


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


torch_logger = TorchOperatorLogger()


