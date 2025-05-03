import json
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from tabulate import tabulate

from precision_compare.analysis.tensor_diff import TensorDiff


class TensorEncoder(json.JSONEncoder):
    """处理Tensor的JSON编码器"""

    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        return super().default(obj)


class AnalysisReport:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_diff_table(self, diffs: List[TensorDiff]):
        """保存差异表格"""
        table_data = []
        for diff in diffs:
            table_data.append(
                [
                    diff.module_name,
                    diff.tensor_type,
                    f"{diff.max_abs_diff:.2e}",
                    f"{diff.max_rel_diff:.2%}",
                    str(diff.location),
                    str(diff.shape),
                    diff.execution_index,
                ]
            )

        headers = [
            "模块",
            "类型",
            "最大绝对误差",
            "最大相对误差",
            "最大差异位置",
            "张量形状",
            "执行顺序",
        ]
        # 同时保存为CSV以便后续分析
        df = pd.DataFrame(table_data, columns=headers)
        df.to_csv(self.output_dir / f"diff_data_{self.timestamp}.csv", index=False)

    def plot_diff_heatmap(self, diffs: List[TensorDiff]):
        """绘制差异热力图"""
        if not diffs:
            # 创建一个空的热力图，显示"无差异"信息
            plt.figure(figsize=(8, 6))
            plt.text(
                0.5,
                0.5,
                "无显著差异",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=20,
            )
            plt.axis("off")
            plt.savefig(
                self.output_dir / f"diff_heatmap_{self.timestamp}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            return

        # 检查是否有足够的数据绘制热图
        if len(diffs) < 2:
            plt.figure(figsize=(12, 6))
            # 只有一个数据点，创建简单的条形图
            module_name = diffs[0].module_name
            plt.bar(
                ["最大绝对误差", "最大相对误差", "余弦相似度"],
                [
                    diffs[0].max_abs_diff,
                    diffs[0].max_rel_diff,
                    diffs[0].cosine_similarity,
                ],
                color=["red", "orange", "blue"],
            )
            plt.title(f"模块 {module_name} ({diffs[0].tensor_type}) 的差异")
            plt.tight_layout()
            plt.savefig(
                self.output_dir / f"diff_heatmap_{self.timestamp}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            return

        # 调整图表大小，根据模块数量增加高度
        module_count = len(diffs)
        # 创建一个新的图形，包含三个子图
        _, (ax1, ax2, ax3) = plt.subplots(
            1, 3, figsize=(24, max(10, module_count * 0.5))
        )

        # 准备热力图数据
        module_names = []
        max_abs_diffs = []
        max_rel_diffs = []
        cosine_diffs = []

        for diff in diffs:
            # 保留完整的模块名，不进行截断
            module_names.append(f"{diff.module_name}\n({diff.tensor_type})")
            max_abs_diffs.append(float(diff.max_abs_diff))
            max_rel_diffs.append(float(diff.max_rel_diff))
            cosine_diffs.append(float(diff.cosine_similarity))

        # 创建最大绝对误差的热力图
        df1 = pd.DataFrame(max_abs_diffs, index=module_names, columns=["最大绝对误差"])
        # 设置绝对误差的阈值为0.1
        norm1 = plt.Normalize(vmin=0, vmax=0.1)
        sns.heatmap(
            df1,
            annot=True,
            fmt=".2e",
            cmap="Reds",
            cbar_kws={"label": "最大绝对误差"},
            norm=norm1,
            ax=ax1,
        )
        ax1.set_title("最大绝对误差热力图")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)

        # 创建最大相对误差的热力图
        df2 = pd.DataFrame(max_rel_diffs, index=module_names, columns=["最大相对误差"])
        # 设置相对误差的阈值为0.01 (1%)
        norm2 = plt.Normalize(vmin=0, vmax=0.01)
        sns.heatmap(
            df2,
            annot=True,
            fmt=".2%",
            cmap="Oranges",
            cbar_kws={"label": "最大相对误差"},
            norm=norm2,
            ax=ax2,
        )
        ax2.set_title("最大相对误差热力图")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)

        # 创建余弦相似度的热力图
        df3 = pd.DataFrame(cosine_diffs, index=module_names, columns=["余弦相似度"])
        # 设置余弦相似度的阈值范围为0.999-1.0
        norm3 = plt.Normalize(vmin=0.999, vmax=1.0)
        sns.heatmap(
            df3,
            annot=True,
            fmt=".4f",
            cmap="Blues_r",
            cbar_kws={"label": "余弦相似度"},
            norm=norm3,
            ax=ax3,
        )
        ax3.set_title("余弦相似度热力图")
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
        ax3.set_yticklabels(ax3.get_yticklabels(), rotation=0)

        # 确保标签可见并调整布局
        plt.tight_layout()

        # 保存图片
        plt.savefig(
            self.output_dir / f"diff_heatmap_{self.timestamp}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print(f"{self.output_dir}/diff_heatmap_{self.timestamp}.png 保存成功")

    def save_summary(self, diffs: List[TensorDiff]):
        """保存分析总结"""
        if not diffs:
            return

        summary = {
            "总体统计": {
                "分析的模块数": len(diffs),
                "最大绝对误差": float(max(d.max_abs_diff for d in diffs)),
                "最大相对误差": float(max(d.max_rel_diff for d in diffs)),
                "最大余弦相似度": float(max(d.cosine_similarity for d in diffs)),
            },
            "按执行顺序的显著差异模块": [
                {
                    "模块名": d.module_name,
                    "类型": d.tensor_type,
                    "最大绝对误差": f"{float(d.max_abs_diff):.2e}",
                    "最大相对误差": f"{float(d.max_rel_diff):.2%}",
                    "最大余弦相似度": f"{float(d.cosine_similarity):.2e}",
                }
                for d in diffs
                if d.cosine_similarity < 0.999
            ][:5],  # 只显示前5个显著差异
        }

        with open(
            self.output_dir / f"analysis_summary_{self.timestamp}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, cls=TensorEncoder)
