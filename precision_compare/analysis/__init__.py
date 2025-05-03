from pathlib import Path
from typing import Dict, Any

from precision_compare.analysis.analyzer import ModelAnalyzer
from precision_compare.analysis.reporter import AnalysisReport


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
    result = {"module_comparisons": {}, "problematic_ops": problematic_ops}

    # 将差异数据转换为字典格式
    for diff in diffs:
        if diff.tensor_type == "outputs":
            result["module_comparisons"][diff.module_name] = {
                "max_abs_diff": diff.max_abs_diff,
                "cosine_similarity": diff.cosine_similarity,
                "shape": diff.shape,
            }

    return result
