import os
import json
import torch
import mindspore as ms
import numpy as np
from safetensors.torch import save_file, load_file
from typing import Dict, Any, Optional, List
from pathlib import Path


class WeightManager:
    """
    权重管理器，用于保存和加载模型权重，提供PyTorch和MindSpore互通的功能
    """

    @staticmethod
    def generate_shared_weights(
        model_config: Dict[str, Any], output_path: str, seed: int = 42
    ) -> str:
        """
        生成统一格式的权重文件，可被PyTorch和MindSpore加载

        Args:
            model_config: 模型配置参数
            output_path: 输出路径
            seed: 随机种子，用于初始化权重

        Returns:
            保存的权重文件路径
        """
        # 设置随机种子确保可重现性
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 模型结构参数
        vocab_size = model_config.get("vocab_size", 32000)
        hidden_size = model_config.get("hidden_size", 768)
        intermediate_size = model_config.get("intermediate_size", 3072)
        num_hidden_layers = model_config.get("num_hidden_layers", 2)
        num_attention_heads = model_config.get("num_attention_heads", 12)

        # 创建权重字典
        weights = {}

        # 嵌入层权重
        weights["model.embed_tokens.weight"] = torch.normal(
            0, 0.02, (vocab_size, hidden_size)
        )

        # 各层权重
        for i in range(num_hidden_layers):
            layer_prefix = f"model.layers.{i}."

            # 自注意力层权重
            head_dim = hidden_size // num_attention_heads
            attn_prefix = layer_prefix + "self_attn."
            weights[attn_prefix + "q_proj.weight"] = torch.normal(
                0, 0.02, (hidden_size, hidden_size)
            )
            weights[attn_prefix + "k_proj.weight"] = torch.normal(
                0, 0.02, (hidden_size, hidden_size)
            )
            weights[attn_prefix + "v_proj.weight"] = torch.normal(
                0, 0.02, (hidden_size, hidden_size)
            )
            weights[attn_prefix + "o_proj.weight"] = torch.normal(
                0, 0.02, (hidden_size, hidden_size)
            )

            # Layer Norm权重
            weights[layer_prefix + "input_layernorm.weight"] = torch.ones(hidden_size)
            weights[layer_prefix + "input_layernorm.bias"] = torch.zeros(hidden_size)
            weights[layer_prefix + "post_attention_layernorm.weight"] = torch.ones(
                hidden_size
            )
            weights[layer_prefix + "post_attention_layernorm.bias"] = torch.zeros(
                hidden_size
            )

            # MLP层权重
            weights[layer_prefix + "mlp.gate_proj.weight"] = torch.normal(
                0, 0.02, (intermediate_size, hidden_size)
            )
            weights[layer_prefix + "mlp.up_proj.weight"] = torch.normal(
                0, 0.02, (intermediate_size, hidden_size)
            )
            weights[layer_prefix + "mlp.down_proj.weight"] = torch.normal(
                0, 0.02, (hidden_size, intermediate_size)
            )

        # 最后的Layer Norm和预测头
        weights["model.norm.weight"] = torch.ones(hidden_size)
        weights["model.norm.bias"] = torch.zeros(hidden_size)
        weights["lm_head.weight"] = torch.normal(0, 0.02, (vocab_size, hidden_size))

        # 保存权重元数据 - 用于不同框架间的参数映射
        metadata = {
            "model_config": model_config,
            "framework_keys": {
                "torch_to_ms": {
                    # 定义PyTorch到MindSpore的键名映射
                    "model.embed_tokens.weight": "model.embed_tokens.weight",
                    "model.norm.weight": "model.norm.weight",
                    "model.norm.bias": "model.norm.bias",
                    "lm_head.weight": "lm_head.weight",
                    # 其他映射关系会在加载过程中根据层次结构自动处理
                }
            },
        }

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 保存为safetensors格式
        save_file(weights, output_path, metadata=None)

        # 同时保存metadata为单独的JSON文件，方便查看
        with open(
            output_path.replace(".safetensors", "_metadata.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"已生成统一权重文件: {output_path}")
        return output_path

    @staticmethod
    def load_torch_weights(model, weights_path: str, strict: bool = True) -> List[str]:
        """
        为PyTorch模型加载权重

        Args:
            model: PyTorch模型实例
            weights_path: 权重文件路径
            strict: 是否严格检查所有键

        Returns:
            未加载的参数列表
        """
        # 加载safetensors文件
        weights = load_file(weights_path)

        # 尝试加载元数据（从单独的JSON文件）
        metadata = None
        metadata_path = weights_path.replace(".safetensors", "_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                print(
                    f"已加载权重元数据, 模型配置包含 {len(metadata.get('model_config', {}))} 项"
                )
            except Exception as e:
                print(f"无法解析权重元数据文件: {str(e)}")

        # 获取模型当前的state_dict
        model_state = model.state_dict()

        # 用于记录未加载的参数
        missing_keys = []

        # 按照模型结构匹配权重
        for name, param in model_state.items():
            if name in weights:
                # 直接加载匹配的权重
                if param.shape == weights[name].shape:
                    param.data.copy_(weights[name])
                else:
                    print(
                        f"警告: 参数 {name} 形状不匹配 - 模型: {param.shape} vs 权重: {weights[name].shape}"
                    )
                    missing_keys.append(name)
            else:
                # 尝试进行名称映射
                found = False
                # 检查name是否以模型的任何前缀开头
                for weight_name in weights:
                    if (
                        weight_name.endswith(name.split(".")[-1])
                        and param.shape == weights[weight_name].shape
                    ):
                        param.data.copy_(weights[weight_name])
                        found = True
                        print(f"映射参数: {weight_name} -> {name}")
                        break

                if not found:
                    missing_keys.append(name)
                    if strict:
                        print(f"缺失参数: {name}, 形状: {param.shape}")

        # 检查是否所有参数都已加载
        if missing_keys and strict:
            print(f"警告: {len(missing_keys)}/{len(model_state)} 参数未加载")
            print(
                f"未加载的参数: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}"
            )

        return missing_keys

    @staticmethod
    def load_mindspore_weights(
        model, weights_path: str, strict: bool = True
    ) -> List[str]:
        """
        为MindSpore模型加载权重

        Args:
            model: MindSpore模型实例
            weights_path: 权重文件路径
            strict: 是否严格检查所有键

        Returns:
            未加载的参数列表
        """
        # 加载safetensors文件
        weights = load_file(weights_path)

        # 尝试加载元数据（从单独的JSON文件）
        metadata = None
        metadata_path = weights_path.replace(".safetensors", "_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                print(
                    f"已加载权重元数据, 模型配置包含 {len(metadata.get('model_config', {}))} 项"
                )
            except Exception as e:
                print(f"无法解析权重元数据文件: {str(e)}")

        # 建立Torch和MindSpore参数名称的映射关系
        torch_to_ms_map = {
            # 基本参数映射
            "weight": "weight",
            "bias": "bias",
            # Embedding层映射
            "embed_tokens.weight": "embed_tokens.embedding_table",
            # LayerNorm映射
            "input_layernorm.weight": "input_layernorm.gamma",
            "input_layernorm.bias": "input_layernorm.beta",
            "post_attention_layernorm.weight": "post_attention_layernorm.gamma",
            "post_attention_layernorm.bias": "post_attention_layernorm.beta",
            "norm.weight": "norm.gamma",
            "norm.bias": "norm.beta",
            # 注意力层映射
            "self_attn.q_proj": "self_attn.q_proj",
            "self_attn.k_proj": "self_attn.k_proj",
            "self_attn.v_proj": "self_attn.v_proj",
            "self_attn.o_proj": "self_attn.o_proj",
            # MLP层映射
            "mlp.gate_proj": "mlp.gate_proj",
            "mlp.up_proj": "mlp.up_proj",
            "mlp.down_proj": "mlp.down_proj",
            # 模型层级映射
            "model.embed_tokens": "model.embed_tokens",
            "model.norm": "model.norm",
            "lm_head": "lm_head",
        }

        # 获取MindSpore模型参数列表
        ms_params = list(model.get_parameters())
        ms_param_dict = {param.name: param for param in ms_params}

        # 用于记录未加载的参数
        missing_keys = []

        # 尝试加载每个MindSpore参数
        for ms_name, ms_param in ms_param_dict.items():
            # 直接匹配
            if ms_name in weights:
                # 转换为NumPy，再转为MindSpore Tensor
                torch_data = weights[ms_name].numpy()
                ms_param.set_data(ms.Tensor(torch_data, ms_param.dtype))
                continue

            # 尝试寻找对应的PyTorch权重名称
            found = False

            # 1. 首先尝试直接匹配完整路径
            for torch_name, torch_weight in weights.items():
                if torch_name == ms_name and ms_param.shape == torch_weight.shape:
                    torch_data = torch_weight.numpy()
                    ms_param.set_data(ms.Tensor(torch_data, ms_param.dtype))
                    found = True
                    print(f"精确匹配参数: {torch_name} -> {ms_name}")
                    break

            # 2. 如果未找到，尝试使用参数映射规则
            if not found:
                for torch_name, torch_weight in weights.items():
                    ms_parts = ms_name.split(".")
                    torch_parts = torch_name.split(".")

                    # 检查层级是否匹配
                    if len(ms_parts) != len(torch_parts):
                        continue

                    # 逐层检查名称映射
                    match = True
                    mapped_torch_name = []

                    for i in range(len(torch_parts)):
                        torch_part = torch_parts[i]
                        ms_part = ms_parts[i]

                        # 检查当前部分是否匹配
                        if torch_part == ms_part:
                            mapped_torch_name.append(torch_part)
                            continue

                        # 检查是否存在直接映射
                        if i > 0:  # 跳过第一层，通常是模型名
                            partial_torch_path = ".".join(torch_parts[i - 1 : i + 1])
                            partial_ms_path = ".".join(ms_parts[i - 1 : i + 1])

                            # 检查部分路径映射
                            if (
                                partial_torch_path in torch_to_ms_map
                                and torch_to_ms_map[partial_torch_path]
                                == partial_ms_path
                            ):
                                mapped_torch_name.append(torch_part)
                                continue

                        # 检查单个部分映射
                        if (
                            torch_part in torch_to_ms_map
                            and torch_to_ms_map[torch_part] == ms_part
                        ):
                            mapped_torch_name.append(torch_part)
                            continue

                        # 检查特殊映射（如embedding_table和gamma/beta）
                        if i == len(torch_parts) - 1:  # 最后一部分是参数名
                            last_two_torch = ".".join(torch_parts[-2:])
                            last_two_ms = ".".join(ms_parts[-2:])

                            if (
                                last_two_torch in torch_to_ms_map
                                and torch_to_ms_map[last_two_torch] == last_two_ms
                            ):
                                mapped_torch_name.append(torch_part)
                                continue

                        # 如果没有匹配
                        match = False
                        break

                    if match and ms_param.shape == torch_weight.shape:
                        try:
                            # 转换为NumPy，再转为MindSpore Tensor
                            torch_data = torch_weight.numpy()
                            ms_param.set_data(ms.Tensor(torch_data, ms_param.dtype))
                            found = True
                            print(f"映射参数: {torch_name} -> {ms_name}")
                            break
                        except Exception as e:
                            print(f"转换参数 {torch_name} -> {ms_name} 失败: {str(e)}")

            # 3. 如果仍未找到，尝试形状匹配的最后机会
            if not found:
                for torch_name, torch_weight in weights.items():
                    # 仅检查形状和最后一个组件名称的部分匹配
                    if ms_param.shape == torch_weight.shape:
                        # 获取最后一层名称
                        ms_last = ms_name.split(".")[-1]
                        torch_last = torch_name.split(".")[-1]

                        # 检查特殊映射关系
                        if (
                            (ms_last == "embedding_table" and torch_last == "weight")
                            or (ms_last == "gamma" and torch_last == "weight")
                            or (ms_last == "beta" and torch_last == "bias")
                        ):
                            try:
                                torch_data = torch_weight.numpy()
                                ms_param.set_data(ms.Tensor(torch_data, ms_param.dtype))
                                found = True
                                print(f"形状匹配参数: {torch_name} -> {ms_name}")
                                break
                            except Exception as e:
                                print(
                                    f"转换参数 {torch_name} -> {ms_name} 失败: {str(e)}"
                                )

            if not found:
                missing_keys.append(ms_name)
                if strict:
                    print(f"缺失参数: {ms_name}, 形状: {ms_param.shape}")

        # 检查是否所有参数都已加载
        if missing_keys and strict:
            print(f"警告: {len(missing_keys)}/{len(ms_param_dict)} 参数未加载")
            print(
                f"未加载的参数: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}"
            )

        return missing_keys

    @staticmethod
    def check_weights_compatibility(
        torch_model, ms_model, weights_path: str
    ) -> Dict[str, Any]:
        """
        检查权重与两个框架模型的兼容性

        Args:
            torch_model: PyTorch模型实例
            ms_model: MindSpore模型实例
            weights_path: 权重文件路径

        Returns:
            兼容性检查结果报告
        """
        # 加载权重
        weights = load_file(weights_path)

        # 获取PyTorch模型参数
        torch_state = torch_model.state_dict()
        torch_params = {name: param.shape for name, param in torch_state.items()}

        # 获取MindSpore模型参数
        ms_params = list(ms_model.get_parameters())
        ms_param_shapes = {param.name: param.shape for param in ms_params}

        # 检查结果
        torch_missing = []
        torch_shape_mismatch = []
        ms_missing = []
        ms_shape_mismatch = []

        # 检查PyTorch参数
        for name, shape in torch_params.items():
            if name not in weights:
                torch_missing.append(name)
            elif shape != tuple(weights[name].shape):
                torch_shape_mismatch.append(
                    {
                        "name": name,
                        "model_shape": shape,
                        "weight_shape": tuple(weights[name].shape),
                    }
                )

        # 检查MindSpore参数
        for name, shape in ms_param_shapes.items():
            if name not in weights:
                ms_missing.append(name)
            elif shape != tuple(weights[name].shape):
                ms_shape_mismatch.append(
                    {
                        "name": name,
                        "model_shape": shape,
                        "weight_shape": tuple(weights[name].shape),
                    }
                )

        # 统计
        total_weights = len(weights)
        torch_total = len(torch_params)
        torch_loadable = torch_total - len(torch_missing) - len(torch_shape_mismatch)
        ms_total = len(ms_param_shapes)
        ms_loadable = ms_total - len(ms_missing) - len(ms_shape_mismatch)

        return {
            "weights_total": total_weights,
            "torch_check": {
                "total_params": torch_total,
                "loadable_params": torch_loadable,
                "loadable_percent": f"{torch_loadable / torch_total * 100:.1f}%",
                "missing_params": len(torch_missing),
                "shape_mismatches": len(torch_shape_mismatch),
                "missing_details": torch_missing[:10],
                "mismatch_details": torch_shape_mismatch[:10],
            },
            "mindspore_check": {
                "total_params": ms_total,
                "loadable_params": ms_loadable,
                "loadable_percent": f"{ms_loadable / ms_total * 100:.1f}%",
                "missing_params": len(ms_missing),
                "shape_mismatches": len(ms_shape_mismatch),
                "missing_details": ms_missing[:10],
                "mismatch_details": ms_shape_mismatch[:10],
            },
        }
