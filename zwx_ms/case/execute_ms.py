import json
import os
from pathlib import Path
from typing import Optional

import mindspore as ms
import numpy as np

import zwx_ms.mock
from zwx_ms.model.llama_ms import create_test_model as create_ms_model
from zwx_ms.mock.mock_ms import register_module_info_ms as register_ms_module

def run_mindspore_model(save_dir: str, error_injector: Optional = None, input_data: np.ndarray = None) -> np.ndarray:
    """运行MindSpore模型并保存结果"""
    # 设置MindSpore上下文
    ms.set_context(mode=ms.PYNATIVE_MODE)

    ms_save_dir = os.path.join(save_dir, "mindspore")
    os.makedirs(ms_save_dir, exist_ok=True)
    weights_dir = os.path.join(save_dir, "weights")

    # 创建模型
    model = create_ms_model()

    # 加载共享权重
    weights_path = os.path.join(weights_dir, "shared_weights.safetensors")
    try:
        model.load_weights(weights_path)
        print("权重加载成功")
    except Exception as e:
        print(f"权重加载失败: {str(e)}")

    # 注入错误（如果有）
    if error_injector:
        print("注入错误到MindSpore模型...")
        error_info = error_injector(model)
        # 保存错误信息
        error_info_path = Path(ms_save_dir) / "injected_error.json"
        with open(error_info_path, "w", encoding="utf-8") as f:
            json.dump(error_info.__dict__, f, indent=2, ensure_ascii=False)

    # 注册模块信息
    register_ms_module(model)

    input_ids = ms.Tensor(input_data, ms.int32)

    # 清除已有日志
    zwx_ms.mock.mock_ms.ms_logger.clear_logs()

    # 前向传播
    outputs = model(input_ids)

    # 保存日志
    zwx_ms.mock.mock_ms.ms_logger.dump_logs(ms_save_dir)

    # 保存输入和输出
    np.save(os.path.join(save_dir, "mindspore_input.npy"), input_ids.asnumpy())
    np.save(os.path.join(save_dir, "mindspore_output.npy"), outputs.asnumpy())

    return outputs.asnumpy()
