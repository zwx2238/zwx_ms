import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import json

from precision_compare.analysis.tensor_diff import TensorDiff
from precision_compare.analysis.font_manager import setup_chinese_font

# 设置中文字体
setup_chinese_font()


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

    def load_tensor(self, path: Path) -> torch.Tensor:
        """加载tensor数据"""
        # 检查文件扩展名
        if str(path).endswith(".npy"):
            # 加载numpy格式文件（MindSpore使用）
            import numpy as np

            # print(f"加载MindSpore格式数据: {path}")
            array = np.load(str(path), allow_pickle=True)
            return torch.from_numpy(array).float()
        else:
            # 加载PyTorch格式文件
            # print(f"加载PyTorch格式数据: {path}")
            return torch.load(str(path))

    def compare_tensors(
        self, t1: torch.Tensor, t2: torch.Tensor
    ) -> Tuple[float, float, Tuple[int, ...], float]:
        """比较两个张量的差异

        Returns:
            max_abs_diff: 最大绝对误差
            cosine_similarity: 余弦相似度
            location: 最大差异的位置
            max_rel_diff: 最大差异位置处的相对误差
        """
        # 确保张量形状一致
        if t1.shape != t2.shape:
            # 如果形状不一致，返回特殊值表示无法比较
            return float("inf"), 0.0, (0,), float("inf")

        # 计算绝对误差
        abs_diff = torch.abs(t1 - t2)

        # 最大绝对误差
        max_diff = torch.max(abs_diff).item()

        # 计算余弦相似度
        t1_flat = t1.flatten()
        t2_flat = t2.flatten()

        # 添加小常数避免零向量
        cosine = torch.nn.functional.cosine_similarity(
            t1_flat.unsqueeze(0), t2_flat.unsqueeze(0), dim=1
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

    def _parse_tensor_file_index(self, file_path: Path, tensor_type: str) -> int:
        """解析张量文件名中的索引

        Args:
            file_path: 文件路径
            tensor_type: 张量类型 (inputs/outputs/parameters)

        Returns:
            文件名中的索引号，如果解析失败返回-1
        """
        try:
            parts = file_path.stem.split("_")
            if tensor_type in parts:
                # 找到tensor_type后的数字
                idx = int(parts[parts.index(tensor_type) + 1])
                return idx
        except (ValueError, IndexError):
            return -1
        return -1

    def _load_tensor_files(
        self, module_path: Path, tensor_type: str, max_files: int = 100
    ) -> List[Path]:
        """加载指定目录下的张量文件

        Args:
            module_path: 模块路径
            tensor_type: 张量类型 (inputs/outputs/parameters)
            max_files: 最大文件数限制

        Returns:
            包含所有找到的文件路径的列表，按照索引排序
        """
        # 获取所有可能的文件
        all_files = list(module_path.glob(f"*{tensor_type}_*.pt")) + list(
            module_path.glob(f"*{tensor_type}_*.npy")
        )
        return all_files

    def _parse_gid_from_filename(self, file_path: Path) -> int:
        """从文件名中解析gid

        Args:
            file_path: 文件路径

        Returns:
            gid，如果不存在则返回0
        """
        try:
            parts = file_path.stem.split("_")
            # 第一部分应该是数字(gid)
            if parts and parts[0].isdigit():
                return int(parts[0])
        except (ValueError, IndexError):
            pass
        return 0

    def _compare_tensor_files(
        self,
        base_files: List[Path],
        ref_files: List[Path],
        module_path: str,
        tensor_type: str,
    ) -> List[TensorDiff]:
        """比较两组张量文件

        Args:
            base_files: 基准文件列表
            ref_files: 参考文件列表
            module_path: 模块路径
            tensor_type: 张量类型

        Returns:
            差异列表
        """
        diffs = []
        min_files = min(len(base_files), len(ref_files))

        if len(base_files) != len(ref_files):
            print(
                f"警告: {module_path} 的基准目录和参考目录的{tensor_type}文件数量不匹配"
            )
            base_files = base_files[:min_files]
            ref_files = ref_files[:min_files]

        for idx, (base_file, ref_file) in enumerate(zip(base_files, ref_files)):
            try:
                base_tensor = self.load_tensor(base_file)
                ref_tensor = self.load_tensor(ref_file)

                if isinstance(base_tensor, torch.Tensor) and isinstance(
                    ref_tensor, torch.Tensor
                ):
                    max_diff, cosine, loc, max_rel_diff = self.compare_tensors(
                        base_tensor, ref_tensor
                    )
                    module_name = (
                        module_path.replace("/", ".")
                        if tensor_type != "parameters"
                        else f"{module_path}/{base_file.stem}"
                    )
                    # 从文件名中获取gid作为execution_index
                    execution_index = self._parse_gid_from_filename(base_file)
                    diffs.append(
                        TensorDiff(
                            module_name,
                            f"{tensor_type}_{idx}"
                            if tensor_type != "parameters"
                            else tensor_type,
                            max_diff,
                            cosine,
                            loc,
                            base_tensor.shape,
                            max_rel_diff,
                            execution_index=execution_index,
                        )
                    )
            except Exception as e:
                print(
                    f"处理模块 {module_path} 的{tensor_type} {base_file.name} 时出错: {str(e)}"
                )

        return diffs

    def compare_module_data(self, module_path: str) -> List[TensorDiff]:
        """比较单个模块的所有数据"""
        base_module_path = self.base_dir / module_path
        ref_module_path = self.ref_dir / module_path
        diffs = []

        # 比较输入和输出
        for tensor_type in ["inputs", "parameters", "outputs"]:
            base_files = self._load_tensor_files(base_module_path, tensor_type)
            ref_files = self._load_tensor_files(ref_module_path, tensor_type)
            diffs.extend(
                self._compare_tensor_files(
                    base_files, ref_files, module_path, tensor_type
                )
            )
        return diffs

    def analyze_all(self, threshold: float = 0) -> List[TensorDiff]:
        """分析所有模块的差异"""
        all_diffs = []

        def process_dir(dir_path: Path, relative_path: str = ""):
            for item in dir_path.iterdir():
                if item.is_dir():
                    new_path = (
                        f"{relative_path}/{item.name}" if relative_path else item.name
                    )
                    # 检查是否存在任何输入输出文件
                    try:
                        module_diffs = self.compare_module_data(new_path)
                        # 只添加超过阈值的差异
                        significant_diffs = [
                            diff
                            for diff in module_diffs
                            if diff.max_abs_diff > threshold
                        ]
                        all_diffs.extend(significant_diffs)
                    except Exception as e:
                        print(f"处理模块 {new_path} 时出错: {str(e)}")
                    process_dir(item, new_path)

        process_dir(self.base_dir)

        all_diffs.sort(key=lambda x: x.execution_index)
        return all_diffs

    def find_problematic_ops(self) -> List[str]:
        """找出可能有问题的算子"""
        diffs = self.analyze_all()
        if not diffs:
            return []

        # 按照执行顺序排序的差异列表
        significant_diffs = [
            d for d in diffs if d.cosine_similarity < 0.999
        ]  # 使用0.999的余弦相似度作为阈值

        if not significant_diffs:
            return []  # 没有显著差异

        # 按模块名分组，查找差异
        module_diffs = {}
        for diff in significant_diffs:
            module_name = (
                diff.module_name.split("/")[0]
                if "/" in diff.module_name
                else diff.module_name
            )
            if module_name not in module_diffs:
                module_diffs[module_name] = []
            module_diffs[module_name].append(diff)

        problematic_ops = []

        # 1. 找到第一个(按执行顺序)出现显著差异的模块
        first_diff = significant_diffs[0]
        problematic_ops.append(
            f"{first_diff.module_name} ({first_diff.tensor_type}, 首次出现显著差异)"
        )

        # 2. 找到差异最大的模块（分别对于inputs/outputs/parameters）
        for tensor_type in ["inputs", "outputs", "parameters"]:
            type_diffs = [d for d in significant_diffs if tensor_type in d.tensor_type]
            if type_diffs:
                max_diff = max(type_diffs, key=lambda d: d.max_abs_diff)
                if max_diff.module_name != first_diff.module_name:
                    problematic_ops.append(
                        f"{max_diff.module_name} ({max_diff.tensor_type}, 最大绝对误差: {max_diff.max_abs_diff:.2e})"
                    )

        # 3. 找到"修复"上游问题的模块
        for module, module_diff_list in module_diffs.items():
            if len(module_diff_list) < 2:
                continue

            # 按执行顺序排序
            sorted_diffs = sorted(module_diff_list, key=lambda x: x.execution_index)

            # 检查参数差异
            param_diffs = [d for d in sorted_diffs if "parameters" in d.tensor_type]
            if param_diffs:
                max_param_diff = max(param_diffs, key=lambda d: d.max_abs_diff)
                problematic_ops.append(
                    f"{max_param_diff.module_name} (参数差异, 最大绝对误差: {max_param_diff.max_abs_diff:.2e})"
                )

            # 检查输入输出差异修复情况
            for i in range(1, len(sorted_diffs)):
                curr = sorted_diffs[i]
                prev = sorted_diffs[i - 1]

                # 如果输入差异大但输出差异小，可能是修复了上游问题
                if (
                    curr.tensor_type == "outputs"
                    and prev.tensor_type == "inputs"
                    and curr.max_abs_diff < prev.max_abs_diff * 0.5
                ):
                    problematic_ops.append(f"{curr.module_name} (可能修复上游问题)")

        # 返回唯一的结果
        return list(dict.fromkeys(problematic_ops))
