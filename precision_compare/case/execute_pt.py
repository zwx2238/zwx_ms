import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import precision_compare.mock
import precision_compare.model
import precision_compare.model.config
from precision_compare.model.llama_pt import LlamaForCausalLM
from precision_compare.model.model import create_test_model
from precision_compare.mock.mock_torch import register_torch_module, TorchOperatorLogger


def run_pytorch_model(
    save_dir: str, error_injector: Optional = None, input_data: np.ndarray = None
) -> np.ndarray:
    """运行PyTorch模型并保存结果"""
    torch_save_dir = os.path.join(save_dir, "torch")
    os.makedirs(torch_save_dir, exist_ok=True)
    weights_dir = os.path.join(save_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # 创建模型
    model = create_test_model(LlamaForCausalLM)

    # 生成共享权重文件（如果不存在）
    weights_path = os.path.join(weights_dir, "shared_weights.safetensors")
    if not os.path.exists(weights_path):
        from precision_compare.utils.weight_utils import WeightManager

        config = precision_compare.model.config.get_llama_config(small=True)
        WeightManager.generate_shared_weights(config, weights_path, seed=42)

    # 加载共享权重
    try:
        model.load_weights(weights_path)
        print("权重加载成功")
    except Exception as e:
        print(f"权重加载失败: {str(e)}")

    # 注入错误（如果有）
    if error_injector:
        print("注入错误到PyTorch模型...")
        error_info = error_injector(model)
        # 保存错误信息
        error_info_path = Path(torch_save_dir) / "injected_error.json"
        with open(error_info_path, "w", encoding="utf-8") as f:
            json.dump(error_info.__dict__, f, indent=2, ensure_ascii=False)

    # 注册模块信息
    torch_logger = TorchOperatorLogger()
    register_torch_module(model, torch_logger)

    # 运行模型
    input_ids = torch.tensor(input_data, dtype=torch.long)

    # 清除已有日志

    # 前向传播
    with torch.no_grad():
        outputs = model(input_ids)

    # 保存日志
    torch_logger.dump_logs(torch_save_dir)

    # 保存输入和输出
    np.save(os.path.join(save_dir, "torch_input.npy"), input_ids.numpy())
    np.save(os.path.join(save_dir, "torch_output.npy"), outputs.detach().numpy())

    return outputs.detach().numpy()
