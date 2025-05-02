import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import json
from matplotlib.font_manager import FontProperties

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


class TensorEncoder(json.JSONEncoder):
    """处理Tensor的JSON编码器"""

    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        return super().default(obj)


@dataclass
class TensorDiff:
    module_name: str
    tensor_type: str  # 'inputs', 'outputs', 或 'parameters'
    max_abs_diff: float
    cosine_similarity: float  # 添加余弦相似度
    location: Tuple[int, ...]  # 最大差异的位置
    shape: Tuple[int, ...]  # tensor的形状
    max_rel_diff: float  # 最大差异位置处的相对误差
    execution_index: int = 0


class AnalysisReport:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_diff_table(self, diffs: List[TensorDiff]):
        """保存差异表格"""
        table_data = []
        for diff in diffs:
            table_data.append([
                diff.module_name,
                diff.tensor_type,
                f"{diff.max_abs_diff:.2e}",
                f"{diff.max_rel_diff:.2%}",
                str(diff.location),
                str(diff.shape),
                diff.execution_index
            ])

        headers = ["模块", "类型", "最大绝对差异", "最大相对差异", "最大差异位置", "张量形状", "执行顺序"]
        table_str = tabulate(table_data, headers=headers, tablefmt="grid")
        
        # 保存到文件
        with open(self.output_dir / f"diff_table_{self.timestamp}.txt", "w", encoding="utf-8") as f:
            f.write(table_str)
        
        # 同时保存为CSV以便后续分析
        df = pd.DataFrame(table_data, columns=headers)
        df.to_csv(self.output_dir / f"diff_data_{self.timestamp}.csv", index=False)

    def plot_diff_heatmap(self, diffs: List[TensorDiff]):
        """绘制差异热力图"""
        if not diffs:
            # 创建一个空的热力图，显示"无差异"信息
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, "无显著差异",
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=20)
            plt.axis('off')
            plt.savefig(self.output_dir / f"diff_heatmap_{self.timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()
            return

        # 检查是否有足够的数据绘制热图
        if len(diffs) < 2:
            plt.figure(figsize=(12, 6))
            # 只有一个数据点，创建简单的条形图
            module_name = diffs[0].module_name
            plt.bar(['最大绝对差异', '最大相对差异', '余弦相似度'],
                   [diffs[0].max_abs_diff, diffs[0].max_rel_diff, diffs[0].cosine_similarity],
                   color=['red', 'orange', 'blue'])
            plt.title(f'模块 {module_name} 的差异')
            plt.tight_layout()
            plt.savefig(self.output_dir / f"diff_heatmap_{self.timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()
            return

        # 调整图表大小，根据模块数量增加高度
        module_count = len(diffs)
        # 创建一个新的图形，包含三个子图
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, max(10, module_count * 0.5)))

        # 准备热力图数据
        module_names = []
        max_abs_diffs = []
        max_rel_diffs = []
        cosine_diffs = []

        for diff in diffs:
            # 保留完整的模块名，不进行截断
            module_names.append(diff.module_name + '.' + diff.tensor_type)
            max_abs_diffs.append(float(diff.max_abs_diff))
            max_rel_diffs.append(float(diff.max_rel_diff))
            cosine_diffs.append(float(diff.cosine_similarity))

        # 创建最大绝对差异的热力图
        df1 = pd.DataFrame(
            max_abs_diffs,
            index=module_names,
            columns=['最大绝对差异']
        )
        sns.heatmap(
            df1,
            annot=True,
            fmt='.2e',
            cmap='Reds',  # 使用红色表示差异大
            cbar_kws={'label': '最大绝对差异'},
            robust=True,
            ax=ax1
        )
        ax1.set_title('最大绝对差异热力图')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)

        # 创建最大相对差异的热力图
        df2 = pd.DataFrame(
            max_rel_diffs,
            index=module_names,
            columns=['最大相对差异']
        )
        sns.heatmap(
            df2,
            annot=True,
            fmt='.2%',  # 用百分比格式显示相对差异
            cmap='Oranges',  # 使用橙色表示差异大
            cbar_kws={'label': '最大相对差异'},
            robust=True,
            ax=ax2
        )
        ax2.set_title('最大相对差异热力图')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)

        # 创建余弦相似度的热力图
        df3 = pd.DataFrame(
            cosine_diffs,
            index=module_names,
            columns=['余弦相似度']
        )
        sns.heatmap(
            df3,
            annot=True,
            fmt='.4f',  # 余弦相似度通常是0-1之间的值，使用小数点格式
            cmap='Blues_r',  # 反转的蓝色图，相似度低的颜色更深
            cbar_kws={'label': '余弦相似度'},
            robust=True,
            ax=ax3
        )
        ax3.set_title('余弦相似度热力图')
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
        ax3.set_yticklabels(ax3.get_yticklabels(), rotation=0)

        # 确保标签可见并调整布局
        plt.tight_layout()

        # 保存图片
        plt.savefig(self.output_dir / f"diff_heatmap_{self.timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_diff_distribution(self, diffs: List[TensorDiff]):
        """绘制差异分布图"""
        if not diffs:
            # 创建一个空的分布图，显示"无差异"信息
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, "无显著差异",
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=20)
            plt.axis('off')
            plt.savefig(self.output_dir / f"diff_distribution_{self.timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()
            return

        # 检查是否有足够的数据绘制分布图
        if len(diffs) < 3:
            # 增大图表尺寸
            plt.figure(figsize=(14, 10))
            # 创建简单的条形图
            labels = []
            abs_values = []
            rel_values = []
            cos_values = []
            
            for diff in diffs:
                # 使用完整模块名
                labels.append(f"{diff.module_name}\n({diff.tensor_type})")
                abs_values.append(diff.max_abs_diff)
                rel_values.append(diff.max_rel_diff)
                cos_values.append(diff.cosine_similarity)

            # 创建一个 2x2 的子图网格
            fig, axs = plt.subplots(2, 2, figsize=(18, 14))
            
            # 绘制绝对差异条形图
            axs[0, 0].bar(range(len(labels)), abs_values, color='red', width=0.6)
            axs[0, 0].set_xticks(range(len(labels)))
            axs[0, 0].set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
            axs[0, 0].set_title('最大绝对差异', fontsize=14)
            axs[0, 0].set_ylabel('差异值', fontsize=12)
            
            # 绘制相对差异条形图
            axs[0, 1].bar(range(len(labels)), rel_values, color='orange', width=0.6)
            axs[0, 1].set_xticks(range(len(labels)))
            axs[0, 1].set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
            axs[0, 1].set_title('最大相对差异', fontsize=14)
            axs[0, 1].set_ylabel('相对差异值', fontsize=12)
            
            # 绘制余弦相似度条形图
            axs[1, 0].bar(range(len(labels)), cos_values, color='blue', width=0.6)
            axs[1, 0].set_xticks(range(len(labels)))
            axs[1, 0].set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
            axs[1, 0].set_title('余弦相似度', fontsize=14)
            axs[1, 0].set_ylabel('相似度', fontsize=12)
            
            # 隐藏最后一个子图
            axs[1, 1].axis('off')
            
            # 调整布局
            plt.tight_layout()
            plt.savefig(self.output_dir / f"diff_distribution_{self.timestamp}.png", dpi=400, bbox_inches='tight')
            plt.close()
            return

        # 对于较多数据点的情况，创建更大更清晰的图表
        plt.figure(figsize=(24, 8))

        # 准备数据，确保所有值都是float类型
        max_abs_diffs = [float(d.max_abs_diff) for d in diffs]
        max_rel_diffs = [float(d.max_rel_diff) for d in diffs]
        cosine_diffs = [float(d.cosine_similarity) for d in diffs]

        # 创建三个子图
        plt.subplot(131)
        plt.hist(max_abs_diffs, bins=min(20, len(diffs)), alpha=0.7, color='skyblue')
        plt.title('最大绝对差异分布', fontsize=14)
        plt.xlabel('差异值', fontsize=12)
        plt.ylabel('频次', fontsize=12)
        plt.grid(axis='y', alpha=0.3)

        plt.subplot(132)
        plt.hist(max_rel_diffs, bins=min(20, len(diffs)), alpha=0.7, color='salmon')
        plt.title('最大相对差异分布', fontsize=14)
        plt.xlabel('相对差异值', fontsize=12)
        plt.ylabel('频次', fontsize=12)
        plt.grid(axis='y', alpha=0.3)

        plt.subplot(133)
        plt.hist(cosine_diffs, bins=min(20, len(diffs)), alpha=0.7, color='lightgreen')
        plt.title('余弦相似度分布', fontsize=14)
        plt.xlabel('相似度', fontsize=12)
        plt.ylabel('频次', fontsize=12)
        plt.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f"diff_distribution_{self.timestamp}.png", dpi=400, bbox_inches='tight')
        plt.close()

    def save_summary(self, diffs: List[TensorDiff]):
        """保存分析总结"""
        if not diffs:
            return

        summary = {
            "总体统计": {
                "分析的模块数": len(diffs),
                "最大绝对差异": float(max(d.max_abs_diff for d in diffs)),
                "最大相对差异": float(max(d.max_rel_diff for d in diffs)),
                "最大余弦相似度": float(max(d.cosine_similarity for d in diffs))
            },
            "按执行顺序的显著差异模块": [
                                            {
                                                "模块名": d.module_name,
                                                "类型": d.tensor_type,
                                                "最大绝对差异": f"{float(d.max_abs_diff):.2e}",
                                                "最大相对差异": f"{float(d.max_rel_diff):.2%}",
                                                "最大余弦相似度": f"{float(d.cosine_similarity):.2e}"
                                            }
                                            for d in diffs if d.cosine_similarity < 0.999
                                        ][:5]  # 只显示前5个显著差异
        }

        with open(self.output_dir / f"analysis_summary_{self.timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, cls=TensorEncoder)

    def save_empty_report(self, empty_diff: TensorDiff, base_dir: str, ref_dir: str):
        """保存空的分析报告"""
        # 保存差异表格
        self.save_diff_table([empty_diff])

        # 创建空的热力图和分布图
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "无显著差异",
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=20)
        plt.axis('off')
        plt.savefig(self.output_dir / f"diff_heatmap_{self.timestamp}.png", dpi=300)
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "无显著差异",
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=20)
        plt.axis('off')
        plt.savefig(self.output_dir / f"diff_distribution_{self.timestamp}.png", dpi=300)
        plt.close()

        # 保存summary
        summary = {
            "总体统计": {
                "分析的模块数": 0,
                "最大绝对差异": 0.0,
                "最大相对差异": 0.0,
                "最大余弦相似度": 1.0
            },
            "基准目录": base_dir,
            "测试目录": ref_dir,
            "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "状态": "无显著差异"
        }

        with open(self.output_dir / f"analysis_summary_{self.timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # 保存问题算子信息（空）
        with open(self.output_dir / "problematic_ops.json", "w", encoding="utf-8") as f:
            json.dump({
                "problematic_ops": [],
                "analysis_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "status": "no_significant_diff",
                "base_dir": base_dir,
                "ref_dir": ref_dir
            }, f, ensure_ascii=False, indent=2, cls=TensorEncoder)


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
            print(f"加载MindSpore格式数据: {path}")
            array = np.load(str(path), allow_pickle=True)
            return torch.from_numpy(array).float()
        else:
            # 加载PyTorch格式文件
            print(f"加载PyTorch格式数据: {path}")
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


def analyze_runs(base_dir: str, ref_dir: str, output_dir: str = None) -> Dict[str, Any]:
    """分析比较两次运行的结果
    
    Args:
        base_dir: 基准运行的日志目录
        ref_dir: 对比运行的日志目录
        output_dir: 输出分析结果的目录，默认为None，会在ref_dir同级创建
        
    Returns:
        Dict[str, Any]: 分析结果字典，包含模块比较等信息
    """
    print(f"分析 {base_dir} 和 {ref_dir} 之间的差异...")

    # 创建分析器
    analyzer = ModelAnalyzer(base_dir, ref_dir)

    # 创建输出目录
    if output_dir is None:
        output_dir = Path(ref_dir).parent / "analysis_results"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"分析结果将保存到: {output_dir}")

    # 创建报告生成器
    report = AnalysisReport(str(output_dir))

    # 分析所有模块
    diffs = analyzer.analyze_all()

    # 保存差异表格
    report.save_diff_table(diffs)

    # 绘制差异热力图
    report.plot_diff_heatmap(diffs)

    # 绘制差异分布图
    report.plot_diff_distribution(diffs)

    # 保存摘要
    report.save_summary(diffs)

    # 找出有问题的操作
    problematic_ops = analyzer.find_problematic_ops()
    print(f"发现 {len(problematic_ops)} 个可能有问题的操作")

    # 可视化问题操作
    if problematic_ops:
        print("可能有问题的操作:")
        for op in problematic_ops:
            print(f"  - {op}")

    print("分析完成!")

    # 准备返回结果
    result = {
        "module_comparisons": {},
        "problematic_ops": problematic_ops
    }

    # 将差异数据转换为字典格式
    for diff in diffs:
        if diff.tensor_type == "outputs":
            result["module_comparisons"][diff.module_name] = {
                "max_abs_diff": diff.max_abs_diff,
                "cosine_similarity": diff.cosine_similarity,
                "shape": diff.shape
            }

    return result


if __name__ == "__main__":
    import click


    @click.command()
    @click.option("--base-dir", type=str, help="基准运行的日志目录")
    @click.option("--ref-dir", type=str, help="对比运行的日志目录")
    def main(base_dir: str, ref_dir: str):
        analyze_runs(base_dir, ref_dir)


    main()
