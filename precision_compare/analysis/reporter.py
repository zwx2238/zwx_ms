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
            plt.title(f"模块 {module_name} 的差异")
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
        fig, (ax1, ax2, ax3) = plt.subplots(
            1, 3, figsize=(24, max(10, module_count * 0.5))
        )

        # 准备热力图数据
        module_names = []
        max_abs_diffs = []
        max_rel_diffs = []
        cosine_diffs = []

        for diff in diffs:
            # 保留完整的模块名，不进行截断
            module_names.append(diff.module_name + "." + diff.tensor_type)
            max_abs_diffs.append(float(diff.max_abs_diff))
            max_rel_diffs.append(float(diff.max_rel_diff))
            cosine_diffs.append(float(diff.cosine_similarity))

        # 创建最大绝对误差的热力图
        df1 = pd.DataFrame(max_abs_diffs, index=module_names, columns=["最大绝对误差"])
        sns.heatmap(
            df1,
            annot=True,
            fmt=".2e",
            cmap="Reds",  # 使用红色表示差异大
            cbar_kws={"label": "最大绝对误差"},
            robust=True,
            ax=ax1,
        )
        ax1.set_title("最大绝对误差热力图")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)

        # 创建最大相对误差的热力图
        df2 = pd.DataFrame(max_rel_diffs, index=module_names, columns=["最大相对误差"])
        sns.heatmap(
            df2,
            annot=True,
            fmt=".2%",  # 用百分比格式显示相对误差
            cmap="Oranges",  # 使用橙色表示差异大
            cbar_kws={"label": "最大相对误差"},
            robust=True,
            ax=ax2,
        )
        ax2.set_title("最大相对误差热力图")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)

        # 创建余弦相似度的热力图
        df3 = pd.DataFrame(cosine_diffs, index=module_names, columns=["余弦相似度"])
        sns.heatmap(
            df3,
            annot=True,
            fmt=".4f",  # 余弦相似度通常是0-1之间的值，使用小数点格式
            cmap="Blues_r",  # 反转的蓝色图，相似度低的颜色更深
            cbar_kws={"label": "余弦相似度"},
            robust=True,
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

    def plot_diff_distribution(self, diffs: List[TensorDiff]):
        """绘制差异分布图"""
        if not diffs:
            # 创建一个空的分布图，显示"无差异"信息
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
                self.output_dir / f"diff_distribution_{self.timestamp}.png",
                dpi=300,
                bbox_inches="tight",
            )
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

            # 绘制绝对误差条形图
            axs[0, 0].bar(range(len(labels)), abs_values, color="red", width=0.6)
            axs[0, 0].set_xticks(range(len(labels)))
            axs[0, 0].set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
            axs[0, 0].set_title("最大绝对误差", fontsize=14)
            axs[0, 0].set_ylabel("差异值", fontsize=12)

            # 绘制相对误差条形图
            axs[0, 1].bar(range(len(labels)), rel_values, color="orange", width=0.6)
            axs[0, 1].set_xticks(range(len(labels)))
            axs[0, 1].set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
            axs[0, 1].set_title("最大相对误差", fontsize=14)
            axs[0, 1].set_ylabel("相对误差值", fontsize=12)

            # 绘制余弦相似度条形图
            axs[1, 0].bar(range(len(labels)), cos_values, color="blue", width=0.6)
            axs[1, 0].set_xticks(range(len(labels)))
            axs[1, 0].set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
            axs[1, 0].set_title("余弦相似度", fontsize=14)
            axs[1, 0].set_ylabel("相似度", fontsize=12)

            # 隐藏最后一个子图
            axs[1, 1].axis("off")

            # 调整布局
            plt.tight_layout()
            plt.savefig(
                self.output_dir / f"diff_distribution_{self.timestamp}.png",
                dpi=400,
                bbox_inches="tight",
            )
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
        plt.hist(max_abs_diffs, bins=min(20, len(diffs)), alpha=0.7, color="skyblue")
        plt.title("最大绝对误差分布", fontsize=14)
        plt.xlabel("差异值", fontsize=12)
        plt.ylabel("频次", fontsize=12)
        plt.grid(axis="y", alpha=0.3)

        plt.subplot(132)
        plt.hist(max_rel_diffs, bins=min(20, len(diffs)), alpha=0.7, color="salmon")
        plt.title("最大相对误差分布", fontsize=14)
        plt.xlabel("相对误差值", fontsize=12)
        plt.ylabel("频次", fontsize=12)
        plt.grid(axis="y", alpha=0.3)

        plt.subplot(133)
        plt.hist(cosine_diffs, bins=min(20, len(diffs)), alpha=0.7, color="lightgreen")
        plt.title("余弦相似度分布", fontsize=14)
        plt.xlabel("相似度", fontsize=12)
        plt.ylabel("频次", fontsize=12)
        plt.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / f"diff_distribution_{self.timestamp}.png",
            dpi=400,
            bbox_inches="tight",
        )
        plt.close()

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

    def save_empty_report(self, empty_diff: TensorDiff, base_dir: str, ref_dir: str):
        """保存空的分析报告"""
        # 保存差异表格
        self.save_diff_table([empty_diff])

        # 创建空的热力图和分布图
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
        plt.savefig(self.output_dir / f"diff_heatmap_{self.timestamp}.png", dpi=300)
        plt.close()

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
            self.output_dir / f"diff_distribution_{self.timestamp}.png", dpi=300
        )
        plt.close()

        # 保存summary
        summary = {
            "总体统计": {
                "分析的模块数": 0,
                "最大绝对误差": 0.0,
                "最大相对误差": 0.0,
                "最大余弦相似度": 1.0,
            },
            "基准目录": base_dir,
            "测试目录": ref_dir,
            "分析时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "状态": "无显著差异",
        }

        with open(
            self.output_dir / f"analysis_summary_{self.timestamp}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # 保存问题算子信息（空）
        with open(self.output_dir / "problematic_ops.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "problematic_ops": [],
                    "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "no_significant_diff",
                    "base_dir": base_dir,
                    "ref_dir": ref_dir,
                },
                f,
                ensure_ascii=False,
                indent=2,
                cls=TensorEncoder,
            )
