import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from precision_compare.case.execute import run_test_case, run_parallel_tests
from datetime import datetime
import click


# 自定义模块导入


@click.command()
@click.option("--parallel", is_flag=True, help="并行执行多个测试")
@click.option(
    "--error-type",
    type=click.Choice(
        [
            "none",
            "weight_noise",
            "dtype_cast",
            "activation_quantization",
            "pynative_graph_switch",
            "tensor_layout",
            "mixed_precision",
        ]
    ),
    default="weight_noise",
    help="错误注入类型",
)
@click.option(
    "--module-path",
    type=str,
    default="model.layers.1.input_layernorm",
    help="注入错误的模块路径",
)
@click.option(
    "--framework",
    type=click.Choice(["torch", "mindspore", "both"]),
    default="both",
    help="要注入错误的框架",
)
@click.option("--scale", type=float, default=0.01, help="权重噪声的比例")
@click.option("--dtype", type=str, default="float16", help="类型转换的目标类型")
@click.option("--bits", type=int, default=4, help="激活量化的位数")
@click.option(
    "--layout",
    type=click.Choice(["NCHW", "NHWC"]),
    default="NCHW",
    help="张量布局格式(用于tensor_layout错误类型)",
)
@click.option(
    "--enable", is_flag=True, help="启用混合精度(用于mixed_precision错误类型)"
)
def main(
    parallel, error_type, module_path, framework, scale, dtype, bits, layout, enable
):
    """测试PyTorch和MindSpore模型精度差异"""

    # 构建测试配置
    test_config = {
        "error_type": error_type,
        "module_path": module_path,
        "framework": framework,
        "scale": scale,
        "dtype": dtype,
        "bits": bits,
        "layout": layout,
        "enable": enable,
    }

    if parallel:
        # 多种测试配置
        test_configs = [
            # 基准比较
            {"error_type": "none", "framework": "both"},
            # MindSpore特有的错误类型
            {
                "error_type": "pynative_graph_switch",
                "module_path": "model.layers.0",
                "framework": "mindspore",
            },
            {
                "error_type": "tensor_layout",
                "module_path": "model.layers.0",
                "framework": "mindspore",
                "layout": "NCHW",
            },
            {
                "error_type": "tensor_layout",
                "module_path": "model.layers.0",
                "framework": "mindspore",
                "layout": "NHWC",
            },
            {
                "error_type": "mixed_precision",
                "module_path": "model",
                "framework": "mindspore",
                "enable": True,
            },
            # # 权重噪声测试
            {
                "error_type": "weight_noise",
                "module_path": "model.layers.1.input_layernorm",
                "framework": "mindspore",
                "scale": 0.01,
            },
            {
                "error_type": "weight_noise",
                "module_path": "lm_head",
                "framework": "torch",
                "scale": 0.01,
            },
            # 类型转换测试
            {
                "error_type": "dtype_cast",
                "module_path": "model",
                "framework": "mindspore",
                "dtype": "float16",
            },
            # 激活量化测试
            {
                "error_type": "activation_quantization",
                "module_path": "model.layers.0",
                "framework": "mindspore",
                "bits": 4,
            },
        ]
        run_parallel_tests(test_configs)
    else:
        # 运行单个测试
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_dir = f"test_case_{error_type}_{timestamp}"
        run_test_case(test_dir, test_config)


if __name__ == "__main__":
    main()
