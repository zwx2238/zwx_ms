import os
import sys
import pytest
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from precision_compare.case.execute import run_test_case, run_parallel_tests


def get_test_dir(error_type):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"test_case_{error_type}_{timestamp}"


def test_base_comparison():
    """基准比较测试"""
    test_config = {"error_type": "none", "framework": "both"}
    run_test_case(get_test_dir("base"), test_config)


def test_weight_noise():
    """权重噪声测试"""
    test_config = {
        "error_type": "weight_noise",
        "module_path": "model.layers.1.input_layernorm",
        "framework": "mindspore",
        "scale": 0.01,
    }
    run_test_case(get_test_dir("weight_noise"), test_config)


def test_dtype_cast():
    """数据类型转换测试"""
    test_config = {
        "error_type": "dtype_cast",
        "module_path": "model",
        "framework": "mindspore",
        "dtype": "float16",
    }
    run_test_case(get_test_dir("dtype_cast"), test_config)


def test_activation_quantization():
    """激活量化测试"""
    test_config = {
        "error_type": "activation_quantization",
        "module_path": "model.layers.0",
        "framework": "mindspore",
        "bits": 4,
    }
    run_test_case(get_test_dir("activation_quantization"), test_config)


def test_pynative_graph_switch():
    """动态图/静态图切换测试"""
    test_config = {
        "error_type": "pynative_graph_switch",
        "module_path": "model.layers.0",
        "framework": "mindspore",
    }
    run_test_case(get_test_dir("pynative_graph_switch"), test_config)


def test_tensor_layout():
    """张量布局测试"""
    test_config = {
        "error_type": "tensor_layout",
        "module_path": "model.layers.0",
        "framework": "mindspore",
        "layout": "NCHW",
    }
    run_test_case(get_test_dir("tensor_layout"), test_config)


def test_mixed_precision():
    """混合精度测试"""
    test_config = {
        "error_type": "mixed_precision",
        "module_path": "model",
        "framework": "mindspore",
        "enable": True,
    }
    run_test_case(get_test_dir("mixed_precision"), test_config)


def test_parallel_execution():
    """并行执行测试"""
    test_configs = [
        {
            "error_type": "weight_noise",
            "module_path": "model.layers.1.input_layernorm",
            "framework": "mindspore",
            "scale": 0.01,
        },
        {
            "error_type": "dtype_cast",
            "module_path": "model",
            "framework": "mindspore",
            "dtype": "float16",
        },
    ]
    run_parallel_tests(test_configs)
