import torch
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.font_manager import FontProperties

from zwx_ms.analysis.tensor_diff import TensorDiff

# 设置中文字体
try:
    # 尝试使用系统中文字体
    font_paths = [
        'C:/Windows/Fonts/SimHei.ttf',  # Windows 黑体
        'C:/Windows/Fonts/Microsoft YaHei UI/msyh.ttc',  # Windows 微软雅黑
        '/System/Library/Fonts/PingFang.ttc',  # macOS
        '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'  # Linux
    ]

    font = None
    for font_path in font_paths:
        try:
            font = FontProperties(fname=font_path)
            break
        except:
            continue

    if font is not None:
        plt.rcParams['font.family'] = font.get_name()
    else:
        print("警告：未找到合适的中文字体，图表中的中文可能无法正确显示")
except:
    print("警告：设置中文字体失败，图表中的中文可能无法正确显示")


class ModelAnalyzer:
    def __init__(self, base_dir: str, ref_dir: str):
        """
        初始化分析器
        :param base_dir: 基准运行的日志目录
        :param ref_dir: 对比运行的日志目录
        """
        self.base_dir = Path(base_dir)
        self.ref_dir = Path(ref_dir)
        self.diffs: List[TensorDiff] = []
        self.execution_order = self._load_execution_order(self.base_dir)

    def _load_execution_order(self, dir_path: Path) -> Dict[str, Dict[str, int]]:
        """加载执行顺序信息"""
        exec_order_path = dir_path / "execution_order.json"
        if not exec_order_path.exists():
            print(f"警告: 未找到执行顺序信息文件 {exec_order_path}")
            return {}

        try:
            with open(exec_order_path, 'r', encoding='utf-8') as f:
                exec_data = json.load(f)
                return exec_data.get('module_execution_index', {})
        except Exception as e:
            print(f"读取执行顺序信息失败: {str(e)}")
            return {}

    def get_execution_index(self, module_name: str, tensor_type: str) -> int:
        """获取模块操作的执行索引"""
        if not self.execution_order:
            return 999999  # 如果没有执行顺序信息，返回一个大数

        # 尝试直接获取执行索引
        module_exec_info = self.execution_order.get(module_name, {})
        if tensor_type in module_exec_info:
            return module_exec_info[tensor_type]

        return 0
        # 如果找不到，使用模块名的最后一部分和张量类型估算
        # 这是一个回退方案，当原始执行顺序信息不可用时使用
        parts = module_name.split('/')

        # 尝试从模块路径中提取层级顺序信息
        layer_order = 500  # 默认中间层
        for part in parts:
            if part.isdigit():
                layer_order = int(part) * 100
            elif any(c.isdigit() for c in part):
                digits = ''.join(c for c in part if c.isdigit())
                if digits:
                    layer_order = int(digits) * 100

        # 为组件类型分配序号
        component_types = {
            "embedding": 10,
            "attn": 20,
            "norm1": 30,
            "norm2": 50,
            "ffn": 40,
            "lm_head": 90
        }

        for comp_type, order in component_types.items():
            if comp_type in parts[-1].lower():
                layer_order += order
                break

        # 张量类型的顺序
        type_order = {"inputs": 0, "parameters": 1, "outputs": 2}
        type_idx = type_order.get(tensor_type, 1) * 10

        return layer_order + type_idx

    def load_tensor(self, path: Path) -> torch.Tensor:
        """加载tensor数据"""
        # 检查文件扩展名
        if str(path).endswith('.npy'):
            # 加载numpy格式文件（MindSpore使用）
            import numpy as np
            # print(f"加载MindSpore格式数据: {path}")
            array = np.load(str(path), allow_pickle=True)
            return torch.from_numpy(array).float()
        else:
            # 加载PyTorch格式文件
            # print(f"加载PyTorch格式数据: {path}")
            return torch.load(str(path))

    def compare_tensors(self, t1: torch.Tensor, t2: torch.Tensor) -> Tuple[float, float, Tuple[int, ...], float]:
        """比较两个张量的差异
        
        Returns:
            max_abs_diff: 最大绝对差异
            cosine_similarity: 余弦相似度
            location: 最大差异的位置
            max_rel_diff: 最大差异位置处的相对误差
        """
        # 确保张量形状一致
        if t1.shape != t2.shape:
            # 如果形状不一致，返回特殊值表示无法比较
            return float('inf'), 0.0, (0,), float('inf')

        # 计算绝对差异
        abs_diff = torch.abs(t1 - t2)

        # 最大绝对差异
        max_diff = torch.max(abs_diff).item()

        # 计算余弦相似度
        t1_flat = t1.flatten()
        t2_flat = t2.flatten()

        # 添加小常数避免零向量
        cosine = torch.nn.functional.cosine_similarity(
            t1_flat.unsqueeze(0),
            t2_flat.unsqueeze(0),
            dim=1
        ).item()

        # 找到最大差异的位置
        max_idx = torch.argmax(abs_diff).item()
        location = np.unravel_index(max_idx, t1.shape)
        
        # 计算最大差异位置处的相对误差
        t1_val = abs(t1[location].item())
        t2_val = abs(t2[location].item())
        max_val = max(t1_val, t2_val)
        max_rel_diff = max_diff / (max_val + 1e-8)  # 添加小常数避免除零

        return max_diff, cosine, location, max_rel_diff

    def compare_module_data(self, module_path: str) -> List[TensorDiff]:
        """比较单个模块的所有数据"""
        base_module_path = self.base_dir / module_path
        ref_module_path = self.ref_dir / module_path

        diffs = []

        # 比较输入
        base_input_file = None
        ref_input_file = None

        # 检查是否存在pt或npy格式的输入文件
        if (base_module_path / "inputs.pt").exists():
            base_input_file = base_module_path / "inputs.pt"
        elif (base_module_path / "inputs.npy").exists():
            base_input_file = base_module_path / "inputs.npy"

        if (ref_module_path / "inputs.pt").exists():
            ref_input_file = ref_module_path / "inputs.pt"
        elif (ref_module_path / "inputs.npy").exists():
            ref_input_file = ref_module_path / "inputs.npy"

        if base_input_file and ref_input_file:
            try:
                base_inputs = self.load_tensor(base_input_file)
                ref_inputs = self.load_tensor(ref_input_file)

                # 处理列表类型的输入
                if isinstance(base_inputs, (list, tuple)):
                    for idx, (base_item, ref_item) in enumerate(zip(base_inputs, ref_inputs)):
                        if isinstance(base_item, torch.Tensor) and isinstance(ref_item, torch.Tensor):
                            max_diff, cosine, loc, max_rel_diff = self.compare_tensors(base_item, ref_item)
                            diffs.append(TensorDiff(
                                f"{module_path.replace('/', '.')}",
                                f"inputs",
                                max_diff,
                                cosine,
                                loc,
                                base_item.shape,
                                max_rel_diff,
                                execution_index=self.get_execution_index(f"{module_path.replace('/', '.')}.inputs", "inputs")
                            ))
                elif isinstance(base_inputs, torch.Tensor) and isinstance(ref_inputs, torch.Tensor):
                    max_diff, cosine, loc, max_rel_diff = self.compare_tensors(base_inputs, ref_inputs)
                    diffs.append(TensorDiff(
                        module_path,
                        "inputs",
                        max_diff,
                        cosine,
                        loc,
                        base_inputs.shape,
                        max_rel_diff,
                        execution_index=self.get_execution_index(f"{module_path.replace('/', '.')}.inputs", "inputs")
                    ))
            except Exception as e:
                print(f"处理模块 {module_path} 的输入时出错: {str(e)}")

        # 比较输出
        base_output_file = None
        ref_output_file = None

        # 检查是否存在pt或npy格式的输出文件
        if (base_module_path / "outputs.pt").exists():
            base_output_file = base_module_path / "outputs.pt"
        elif (base_module_path / "outputs.npy").exists():
            base_output_file = base_module_path / "outputs.npy"

        if (ref_module_path / "outputs.pt").exists():
            ref_output_file = ref_module_path / "outputs.pt"
        elif (ref_module_path / "outputs.npy").exists():
            ref_output_file = ref_module_path / "outputs.npy"

        if base_output_file and ref_output_file:
            try:
                base_outputs = self.load_tensor(base_output_file)
                ref_outputs = self.load_tensor(ref_output_file)

                # 处理列表类型的输出
                if isinstance(base_outputs, (list, tuple)):
                    for idx, (base_item, ref_item) in enumerate(zip(base_outputs, ref_outputs)):
                        if isinstance(base_item, torch.Tensor) and isinstance(ref_item, torch.Tensor):
                            max_diff, cosine, loc, max_rel_diff = self.compare_tensors(base_item, ref_item)
                            diffs.append(TensorDiff(
                                f"{module_path.replace('/', '.')}",
                                "outputs",
                                max_diff,
                                cosine,
                                loc,
                                base_item.shape,
                                max_rel_diff,
                                execution_index=self.get_execution_index(f"{module_path.replace('/', '.')}.outputs", "outputs")
                            ))
                elif isinstance(base_outputs, torch.Tensor) and isinstance(ref_outputs, torch.Tensor):
                    max_diff, cosine, loc, max_rel_diff = self.compare_tensors(base_outputs, ref_outputs)
                    diffs.append(TensorDiff(
                        module_path,
                        "outputs",
                        max_diff,
                        cosine,
                        loc,
                        base_outputs.shape,
                        max_rel_diff,
                        execution_index=self.get_execution_index(f"{module_path.replace('/', '.')}.outputs", "outputs")
                    ))
            except Exception as e:
                print(f"处理模块 {module_path} 的输出时出错: {str(e)}")

        # 比较参数 - 支持不同的文件扩展名
        # 需要处理.pt和.npy两种文件格式
        base_param_files = list(base_module_path.glob("*.pt")) + list(base_module_path.glob("*.npy"))
        for param_file in base_param_files:
            if param_file.name in ["inputs.pt", "outputs.pt", "inputs.npy", "outputs.npy"]:
                continue

            # 检查在ref_module_path中是否有相同名称但可能不同扩展名的文件
            param_stem = param_file.stem
            ref_pt_file = ref_module_path / f"{param_stem}.pt"
            ref_npy_file = ref_module_path / f"{param_stem}.npy"

            ref_param_file = None
            if ref_pt_file.exists():
                ref_param_file = ref_pt_file
            elif ref_npy_file.exists():
                ref_param_file = ref_npy_file

            if not ref_param_file:
                continue

            try:
                base_param = self.load_tensor(param_file)
                ref_param = self.load_tensor(ref_param_file)
                max_diff, cosine, loc, max_rel_diff = self.compare_tensors(base_param, ref_param)
                diffs.append(TensorDiff(
                    f"{module_path}/{param_stem}",
                    "parameters",
                    max_diff,
                    cosine,
                    loc,
                    base_param.shape,
                    max_rel_diff,
                    execution_index=self.get_execution_index(f"{module_path}/{param_stem}", "parameters")
                ))
            except Exception as e:
                print(f"处理模块 {module_path} 的参数 {param_file.name} 时出错: {str(e)}")

        return diffs

    def analyze_all(self, threshold: float = 0) -> List[TensorDiff]:
        """分析所有模块的差异"""
        all_diffs = []

        def process_dir(dir_path: Path, relative_path: str = ""):
            for item in dir_path.iterdir():
                if item.is_dir():
                    new_path = f"{relative_path}/{item.name}" if relative_path else item.name
                    # 如果目录包含 inputs.pt/inputs.npy 或 outputs.pt/outputs.npy，说明是一个模块目录
                    if ((item / "inputs.pt").exists() or (item / "outputs.pt").exists() or
                            (item / "inputs.npy").exists() or (item / "outputs.npy").exists()):
                        try:
                            module_diffs = self.compare_module_data(new_path)
                            # 只添加超过阈值的差异
                            significant_diffs = [diff for diff in module_diffs if diff.max_abs_diff > threshold]
                            all_diffs.extend(significant_diffs)
                        except Exception as e:
                            print(f"处理模块 {new_path} 时出错: {str(e)}")
                    process_dir(item, new_path)

        process_dir(self.base_dir)

        # 计算并添加执行索引
        for diff in all_diffs:
            diff.execution_index = self.get_execution_index(diff.module_name, diff.tensor_type)

        # 使用执行顺序信息排序
        all_diffs.sort(key=lambda x: x.execution_index)

        return all_diffs

    def find_problematic_ops(self) -> List[str]:
        """找出可能有问题的算子"""
        diffs = self.analyze_all()
        if not diffs:
            return []

        # 按照执行顺序排序的差异列表
        significant_diffs = [d for d in diffs if d.cosine_similarity < 0.999]  # 使用1%的相对差异作为阈值

        if not significant_diffs:
            return []  # 没有显著差异

        # 按模块名分组，查找差异
        module_diffs = {}
        for diff in significant_diffs:
            module_name = diff.module_name.split('/')[0] if '/' in diff.module_name else diff.module_name
            if module_name not in module_diffs:
                module_diffs[module_name] = []
            module_diffs[module_name].append(diff)

        problematic_ops = []

        # 1. 找到第一个(按执行顺序)出现显著差异的模块
        first_diff = significant_diffs[0]
        problematic_ops.append(f"{first_diff.module_name} (首次出现显著差异)")

        # 2. 找到差异最大的模块
        max_diff = max(significant_diffs, key=lambda d: d.cosine_similarity)
        if max_diff.module_name != first_diff.module_name:
            problematic_ops.append(f"{max_diff.module_name} (最大余弦相似度: {max_diff.cosine_similarity:.2%})")

        # 3. 找到"修复"上游问题的模块
        for module, module_diff_list in module_diffs.items():
            if len(module_diff_list) < 2:
                continue

            # 按执行顺序排序
            sorted_diffs = sorted(module_diff_list,
                                  key=lambda d: self.get_execution_index(d.module_name, d.tensor_type))

            for i in range(1, len(sorted_diffs)):
                curr = sorted_diffs[i]
                prev = sorted_diffs[i - 1]

                # 如果输入差异大但输出差异小，可能是修复了上游问题
                if (curr.tensor_type == "outputs" and
                        prev.tensor_type == "inputs" and
                        curr.max_abs_diff < prev.max_abs_diff * 0.5):
                    problematic_ops.append(f"{curr.module_name} (可能修复上游问题)")

        # 返回唯一的结果
        return list(dict.fromkeys(problematic_ops))


