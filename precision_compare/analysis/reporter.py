import json
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from tabulate import tabulate
import numpy as np

from precision_compare.analysis.tensor_diff import TensorDiff


class TensorEncoder(json.JSONEncoder):
    """处理Tensor的JSON编码器"""

    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)


class AnalysisReport:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_diff_table(self, df: pd.DataFrame):
        """保存差异表格"""
        # 重命名列以使用中文标题
        headers = {
            "module_name": "模块",
            "tensor_type": "类型",
            "max_abs_diff": "最大绝对误差",
            "max_rel_diff": "最大相对误差",
            "location": "最大差异位置",
            "shape": "张量形状",
            "execution_index": "执行顺序",
        }
        df = df.rename(columns=headers)

        # 格式化数值列
        df["最大绝对误差"] = df["最大绝对误差"].apply(lambda x: f"{x:.2e}")
        df["最大相对误差"] = df["最大相对误差"].apply(lambda x: f"{x:.2%}")

        # 保存为CSV
        df.to_csv(self.output_dir / f"diff_data_{self.timestamp}.csv", index=False)

    def plot_diff_heatmap(self, df: pd.DataFrame):
        """绘制差异热力图"""
        if df.empty:
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
        if len(df) < 2:
            plt.figure(figsize=(12, 6))
            # 只有一个数据点，创建简单的条形图
            module_name = df.iloc[0]["module_name"]
            plt.bar(
                ["最大绝对误差", "最大相对误差", "余弦相似度"],
                [
                    df.iloc[0]["max_abs_diff"],
                    df.iloc[0]["max_rel_diff"],
                    df.iloc[0]["cosine_similarity"],
                ],
                color=["red", "orange", "blue"],
            )
            plt.title(f"模块 {module_name} ({df.iloc[0]['tensor_type']}) 的差异")
            plt.tight_layout()
            plt.savefig(
                self.output_dir / f"diff_heatmap_{self.timestamp}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            return

        # 调整图表大小，根据模块数量增加高度
        module_count = len(df)
        # 创建一个新的图形，包含三个子图
        _, (ax1, ax2, ax3) = plt.subplots(
            1, 3, figsize=(24, max(10, module_count * 0.5))
        )

        # 准备热力图数据
        pivot_abs = pd.pivot_table(
            df,
            values="max_abs_diff",
            index="module_name",
            columns="tensor_type",
            aggfunc="max",
        )
        pivot_rel = pd.pivot_table(
            df,
            values="max_rel_diff",
            index="module_name",
            columns="tensor_type",
            aggfunc="max",
        )
        pivot_cos = pd.pivot_table(
            df,
            values="cosine_similarity",
            index="module_name",
            columns="tensor_type",
            aggfunc="max",
        )

        # 绘制最大绝对误差热力图
        norm1 = plt.Normalize(vmin=0, vmax=pivot_abs.max().max())
        sns.heatmap(
            pivot_abs,
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

        # 绘制最大相对误差热力图
        norm2 = plt.Normalize(vmin=0, vmax=pivot_rel.max().max())
        sns.heatmap(
            pivot_rel,
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

        # 绘制余弦相似度热力图
        norm3 = plt.Normalize(vmin=pivot_cos.min().min(), vmax=1)
        sns.heatmap(
            pivot_cos,
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

        plt.tight_layout()
        plt.savefig(
            self.output_dir / f"diff_heatmap_{self.timestamp}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print(f"{self.output_dir}/diff_heatmap_{self.timestamp}.png 保存成功")

    def save_summary(self, df: pd.DataFrame):
        """保存分析总结"""
        if df.empty:
            return

        summary = {
            "总体统计": {
                "分析的模块数": len(df),
                "最大绝对误差": float(df["max_abs_diff"].max()),
                "最大相对误差": float(df["max_rel_diff"].max()),
                "最大余弦相似度": float(df["cosine_similarity"].max()),
            },
            "按执行顺序的显著差异模块": df[df["cosine_similarity"] < 0.999]
            .head(5)
            .apply(
                lambda row: {
                    "模块名": row["module_name"],
                    "类型": row["tensor_type"],
                    "最大绝对误差": f"{float(row['max_abs_diff']):.2e}",
                    "最大相对误差": f"{float(row['max_rel_diff']):.2%}",
                    "最大余弦相似度": f"{float(row['cosine_similarity']):.2e}",
                },
                axis=1,
            )
            .tolist(),
        }

        with open(
            self.output_dir / f"analysis_summary_{self.timestamp}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, cls=TensorEncoder)
