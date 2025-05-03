import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Literal

import numpy as np
import torch
import mindspore as ms

from precision_compare.analysis import analyze_runs
from precision_compare.mock.error import ErrorInjector
from precision_compare.model.model import create_test_model
from precision_compare.model.llama_ms import LlamaForCausalLM as MSLlamaForCausalLM
from precision_compare.model.llama_pt import LlamaForCausalLM as PTLlamaForCausalLM
from precision_compare.mock.mock_ms import register_ms_module, MindSporeOperatorLogger
from precision_compare.mock.mock_torch import register_torch_module, TorchOperatorLogger
from precision_compare.utils import set_random_seed
from precision_compare.utils.weight_utils import WeightManager
from precision_compare.model.config import get_llama_config


def run_model(
    save_dir: str,
    framework: Literal["torch", "mindspore"],
    error_injector: Optional = None,
    input_data: np.ndarray = None,
) -> np.ndarray:
    """统一的模型执行函数

    Args:
        save_dir: 保存目录
        framework: 使用的框架，可选 "torch" 或 "mindspore"
        error_injector: 错误注入器
        input_data: 输入数据

    Returns:
        模型输出的numpy数组
    """
    # 创建框架特定的保存目录
    framework_save_dir = os.path.join(save_dir, framework)
    os.makedirs(framework_save_dir, exist_ok=True)
    weights_dir = os.path.join(save_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # 根据框架选择模型类和设置
    if framework == "torch":
        model_class = PTLlamaForCausalLM
        logger_class = TorchOperatorLogger
        register_module = register_torch_module
        input_tensor = torch.tensor(input_data, dtype=torch.long)
    else:  # mindspore
        ms.set_context(mode=ms.PYNATIVE_MODE)
        model_class = MSLlamaForCausalLM
        logger_class = MindSporeOperatorLogger
        register_module = register_ms_module
        input_tensor = ms.Tensor(input_data, ms.int32)

    # 创建模型
    model = create_test_model(model_class)

    # 处理权重
    weights_path = os.path.join(weights_dir, "shared_weights.safetensors")
    if not os.path.exists(weights_path) and framework == "torch":
        config = get_llama_config(small=True)
        WeightManager.generate_shared_weights(config, weights_path, seed=42)

    # 加载权重
    try:
        model.load_weights(weights_path)
        print("权重加载成功")
    except Exception as e:
        print(f"权重加载失败: {str(e)}")

    # 注入错误（如果有）
    if error_injector:
        print(f"注入错误到{framework}模型...")
        error_info = error_injector(model)
        # 保存错误信息
        error_info_path = Path(framework_save_dir) / "injected_error.json"
        with open(error_info_path, "w", encoding="utf-8") as f:
            json.dump(error_info.__dict__, f, indent=2, ensure_ascii=False)

    # 注册模块信息
    logger = logger_class()
    register_module(model, logger)

    # 执行前向传播
    if framework == "torch":
        with torch.no_grad():
            outputs = model(input_tensor)
        result = outputs.detach().numpy()
    else:
        outputs = model(input_tensor)
        result = outputs.asnumpy()

    # 保存日志
    logger.dump_logs(framework_save_dir)

    # 保存输入和输出
    np.save(
        os.path.join(save_dir, f"{framework}_input.npy"),
        input_tensor.numpy() if framework == "torch" else input_tensor.asnumpy(),
    )
    np.save(os.path.join(save_dir, f"{framework}_output.npy"), result)

    return result


def run_test_case(test_dir: str, config: Dict) -> None:
    """运行单个测试用例"""
    test_dir_path = Path(test_dir)
    print(f"\n运行测试用例: {test_dir_path.name}")

    # 设置随机种子
    set_random_seed(42)

    # 创建测试目录
    os.makedirs(test_dir, exist_ok=True)

    # 保存测试配置
    with open(os.path.join(test_dir, "test_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    input_data = get_input_ids()

    # 分别运行PyTorch和MindSpore模型
    print("运行PyTorch模型...")
    torch_error_injector = ErrorInjector.create_error_injector(config, "torch")
    torch_output = run_model(test_dir, "torch", torch_error_injector, input_data)

    print("\n运行MindSpore模型...")
    ms_error_injector = ErrorInjector.create_error_injector(config, "mindspore")
    ms_output = run_model(test_dir, "mindspore", ms_error_injector, input_data)

    # 比较两个模型的输出
    print("\n比较两个框架的输出差异...")
    comparison = compare_outputs(torch_output, ms_output)

    # 保存比较结果
    with open(
        os.path.join(test_dir, "comparison_results.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    # 打印主要差异指标
    print("\n===== 精度对比关键指标 =====")
    print(f"输入形状: {comparison.get('torch_shape')}")
    print(
        f"PyTorch dtype: {comparison.get('torch_dtype')}, MindSpore dtype: {comparison.get('ms_dtype')}"
    )
    print(f"最大绝对误差: {comparison.get('max_abs_diff'):.6e}")
    print(f"余弦相似度: {comparison.get('cosine_similarity'):.6f}")
    print(
        f"最大误差位置 {comparison.get('max_diff_location')} 的相对误差: {comparison.get('rel_err_at_max_diff'):.6%}"
    )
    print(
        f"最大误差位置的值 - PyTorch: {comparison.get('torch_value_at_max'):.6f}, MindSpore: {comparison.get('ms_value_at_max'):.6f}"
    )
    print("===========================")

    # 分析模型内部差异
    print("\n开始详细分析内部差异...")
    try:
        result = analyze_runs(
            os.path.join(test_dir, "torch"),
            os.path.join(test_dir, "mindspore"),
            os.path.join(test_dir, "analysis"),
        )

        # 打印分析结果摘要
        print("\n===== 内部层分析摘要 =====")
        if "module_comparisons" in result:
            modules = result["module_comparisons"]
            if modules:
                max_diff_module = max(
                    modules.items(), key=lambda x: x[1].get("max_abs_diff", 0)
                )
                print(
                    f"最大差异模块: {max_diff_module[0]}, 绝对误差: {max_diff_module[1].get('max_abs_diff', 0):.6e}"
                )

                # 输出余弦相似度最小的模块
                min_cosine_module = min(
                    modules.items(), key=lambda x: x[1].get("cosine_similarity", 1.0)
                )
                print(
                    f"余弦相似度最低模块: {min_cosine_module[0]}, 相似度: {min_cosine_module[1].get('cosine_similarity', 0):.6f}"
                )
            else:
                print("未找到模块比较结果")
        else:
            print("分析结果未包含模块比较信息")
        print("=========================")

        print(f"详细分析结果已保存到 {os.path.join(test_dir, 'analysis')}")
    except Exception as e:
        print(f"分析过程出错: {str(e)}")
        import traceback

        traceback.print_exc()


def get_input_ids():
    # 首先生成固定的输入数据，确保两个框架使用相同的输入
    print("生成统一输入数据...")
    batch_size = 1
    seq_len = 10
    # 使用固定种子生成随机输入
    np.random.seed(42)
    input_data = np.random.randint(0, 32000, (batch_size, seq_len))
    return input_data


def run_parallel_tests(test_configs: List[Dict]) -> None:
    """并行运行多个测试用例"""
    print(f"\n开始执行 {len(test_configs)} 个测试用例...")

    # 创建测试用例目录
    test_cases = []
    for idx, config in enumerate(test_configs):
        case_name = f"case{idx + 1}"
        test_dir = create_test_case(case_name, config)
        test_cases.append((test_dir, config))
        print(f"创建测试用例: {Path(test_dir).name}")

    # 使用进程池并行执行测试
    with ProcessPoolExecutor(
        max_workers=min(len(test_cases), os.cpu_count() or 1)
    ) as executor:
        # 提交所有测试任务
        futures = []
        for test_dir, config in test_cases:
            future = executor.submit(run_test_case, test_dir, config)
            futures.append(future)

        # 等待所有任务完成
        for future in futures:
            try:
                future.result()
            except Exception:
                import traceback

                traceback.print_exc()

    print("\n所有测试完成!")


def create_test_case(case_name: str, config: Dict) -> str:
    """创建测试用例目录"""
    # 获取错误类型
    error_type = config.get("error_type", "none")

    # 使用日期后缀确保唯一性
    date_suffix = datetime.now().strftime("%Y%m%d")

    # 创建目录名
    dir_name = f"{case_name}_{error_type}_{date_suffix}"
    base_dir = Path("test_cases") / dir_name

    # 创建测试目录
    base_dir.mkdir(parents=True, exist_ok=True)

    # 创建分析结果目录
    analysis_dir = base_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    # 保存错误配置
    with open(base_dir / "test_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    return str(base_dir)


def compare_outputs(torch_output: np.ndarray, ms_output: np.ndarray) -> Dict:
    """比较两个框架的输出差异"""
    # 形状和数据类型信息
    torch_shape = torch_output.shape
    ms_shape = ms_output.shape
    torch_dtype = str(torch_output.dtype)
    ms_dtype = str(ms_output.dtype)

    # 检查形状是否匹配
    if torch_shape != ms_shape:
        return {
            "error": f"形状不匹配: PyTorch {torch_shape} vs MindSpore {ms_shape}",
            "torch_shape": torch_shape,
            "ms_shape": ms_shape,
            "torch_dtype": torch_dtype,
            "ms_dtype": ms_dtype,
        }

    # 计算差异
    diff = np.abs(torch_output - ms_output)

    # 1. 最大绝对误差
    max_diff = float(np.max(diff))

    # 2. 找到最大差异的位置
    max_idx = np.unravel_index(np.argmax(diff), diff.shape)
    torch_value = float(torch_output[max_idx])
    ms_value = float(ms_output[max_idx])

    # 3. 最大差异位置的相对误差
    rel_err_at_max = float(
        diff[max_idx] / (max(abs(torch_value), abs(ms_value)) + 1e-10)
    )

    # 4. 余弦相似度
    # 将数组展平以计算余弦相似度
    torch_flat = torch_output.flatten()
    ms_flat = ms_output.flatten()

    # 计算余弦相似度 = (a·b) / (||a|| * ||b||)
    dot_product = np.sum(torch_flat * ms_flat)
    torch_norm = np.sqrt(np.sum(torch_flat**2))
    ms_norm = np.sqrt(np.sum(ms_flat**2))

    cosine_similarity = float(dot_product / (torch_norm * ms_norm + 1e-10))

    # 返回关键指标
    return {
        # 形状和数据类型信息
        "torch_shape": torch_shape,
        "ms_shape": ms_shape,
        "torch_dtype": torch_dtype,
        "ms_dtype": ms_dtype,
        # 关键指标
        "max_abs_diff": max_diff,
        "cosine_similarity": cosine_similarity,
        "rel_err_at_max_diff": rel_err_at_max,
        # 最大差异的位置信息
        "max_diff_location": [int(i) for i in max_idx],
        "torch_value_at_max": torch_value,
        "ms_value_at_max": ms_value,
    }
