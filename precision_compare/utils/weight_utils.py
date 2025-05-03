import os
import json
import torch
import mindspore as ms
import numpy as np
from safetensors.torch import save_file, load_file
from typing import Dict, Any, Optional, List, Tuple, NamedTuple
from pathlib import Path


class MatchResult(NamedTuple):
    """参数匹配结果"""

    torch_data: Optional[np.ndarray]
    torch_name: Optional[str]
    match_type: str


class WeightLoadResult(NamedTuple):
    """权重加载结果"""

    missing_keys: List[str]
    loaded_count: int
    total_count: int


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
    def _get_torch_to_ms_map() -> Dict[str, str]:
        """获取PyTorch到MindSpore的参数名映射关系"""
        return {
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

    @staticmethod
    def _try_load_metadata(weights_path: str) -> Optional[Dict]:
        """尝试加载权重元数据"""
        metadata_path = weights_path.replace(".safetensors", "_metadata.json")
        if not os.path.exists(metadata_path):
            return None

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            print(
                f"已加载权重元数据, 模型配置包含 {len(metadata.get('model_config', {}))} 项"
            )
            return metadata
        except Exception as e:
            print(f"无法解析权重元数据文件: {str(e)}")
            return None

    @staticmethod
    def _try_direct_match(
        ms_name: str, ms_param: ms.Parameter, weights: Dict
    ) -> MatchResult:
        """尝试直接匹配参数"""
        if ms_name in weights and ms_param.shape == weights[ms_name].shape:
            return MatchResult(weights[ms_name].numpy(), ms_name, "direct")
        return MatchResult(None, None, "")

    @staticmethod
    def _check_partial_match(
        torch_part: str,
        ms_part: str,
        torch_parts: List[str],
        ms_parts: List[str],
        i: int,
        torch_to_ms_map: Dict[str, str],
    ) -> bool:
        """检查部分名称匹配"""
        # 直接匹配
        if torch_part == ms_part:
            return True

        # 检查映射
        if i > 0:
            partial_torch = ".".join(torch_parts[i - 1 : i + 1])
            partial_ms = ".".join(ms_parts[i - 1 : i + 1])
            if (
                partial_torch in torch_to_ms_map
                and torch_to_ms_map[partial_torch] == partial_ms
            ):
                return True

        # 单个部分映射
        if torch_part in torch_to_ms_map and torch_to_ms_map[torch_part] == ms_part:
            return True

        return False

    @staticmethod
    def _check_special_match(
        torch_parts: List[str],
        ms_parts: List[str],
        i: int,
        torch_to_ms_map: Dict[str, str],
    ) -> bool:
        """检查特殊映射匹配"""
        if i == len(torch_parts) - 1:
            last_two_torch = ".".join(torch_parts[-2:])
            last_two_ms = ".".join(ms_parts[-2:])
            if (
                last_two_torch in torch_to_ms_map
                and torch_to_ms_map[last_two_torch] == last_two_ms
            ):
                return True
        return False

    @staticmethod
    def _check_name_match(
        ms_parts: List[str], torch_parts: List[str], torch_to_ms_map: Dict[str, str]
    ) -> bool:
        """检查参数名称是否匹配"""
        if len(ms_parts) != len(torch_parts):
            return False

        for i in range(len(torch_parts)):
            # 检查常规匹配
            if WeightManager._check_partial_match(
                torch_parts[i], ms_parts[i], torch_parts, ms_parts, i, torch_to_ms_map
            ):
                continue

            # 检查特殊匹配
            if WeightManager._check_special_match(
                torch_parts, ms_parts, i, torch_to_ms_map
            ):
                continue

            return False
        return True

    @staticmethod
    def _try_name_mapping_match(
        ms_name: str,
        ms_param: ms.Parameter,
        weights: Dict,
        torch_to_ms_map: Dict[str, str],
    ) -> MatchResult:
        """尝试通过名称映射匹配参数"""
        ms_parts = ms_name.split(".")

        for torch_name, torch_weight in weights.items():
            if ms_param.shape != torch_weight.shape:
                continue

            torch_parts = torch_name.split(".")
            if WeightManager._check_name_match(ms_parts, torch_parts, torch_to_ms_map):
                return MatchResult(torch_weight.numpy(), torch_name, "mapping")

        return MatchResult(None, None, "")

    @staticmethod
    def _try_shape_match(
        ms_name: str, ms_param: ms.Parameter, weights: Dict
    ) -> MatchResult:
        """尝试通过形状匹配参数"""
        ms_last = ms_name.split(".")[-1]

        for torch_name, torch_weight in weights.items():
            if ms_param.shape != torch_weight.shape:
                continue

            torch_last = torch_name.split(".")[-1]
            if WeightManager._check_shape_match_rules(ms_last, torch_last):
                return MatchResult(torch_weight.numpy(), torch_name, "shape")

        return MatchResult(None, None, "")

    @staticmethod
    def _check_shape_match_rules(ms_last: str, torch_last: str) -> bool:
        """检查形状匹配规则"""
        return any(
            [
                ms_last == "embedding_table" and torch_last == "weight",
                ms_last == "gamma" and torch_last == "weight",
                ms_last == "beta" and torch_last == "bias",
            ]
        )

    @staticmethod
    def _try_load_parameter(
        ms_param: ms.Parameter,
        torch_data: np.ndarray,
        match_type: str,
        torch_name: Optional[str] = None,
    ) -> bool:
        """尝试加载参数"""
        try:
            ms_param.set_data(ms.Tensor(torch_data, ms_param.dtype))
            match_desc = {
                "direct": "精确匹配参数",
                "mapping": "映射参数",
                "shape": "形状匹配参数",
            }
            msg = match_desc.get(match_type, "")
            if msg:
                if torch_name and torch_name != ms_param.name:
                    print(f"{msg}: {torch_name} -> {ms_param.name}")
                else:
                    print(f"{msg}: {ms_param.name}")
            return True
        except Exception as e:
            print(f"加载参数 {ms_param.name} 失败: {str(e)}")
            return False

    @staticmethod
    def _match_parameter(
        ms_param: ms.Parameter, weights: Dict, torch_to_ms_map: Dict[str, str]
    ) -> bool:
        """匹配并加载单个参数"""
        ms_name = ms_param.name

        # 1. 直接匹配
        match_result = WeightManager._try_direct_match(ms_name, ms_param, weights)
        if match_result.torch_data is not None:
            return WeightManager._try_load_parameter(
                ms_param,
                match_result.torch_data,
                match_result.match_type,
                match_result.torch_name,
            )

        # 2. 名称映射匹配
        match_result = WeightManager._try_name_mapping_match(
            ms_name, ms_param, weights, torch_to_ms_map
        )
        if match_result.torch_data is not None:
            return WeightManager._try_load_parameter(
                ms_param,
                match_result.torch_data,
                match_result.match_type,
                match_result.torch_name,
            )

        # 3. 形状匹配
        match_result = WeightManager._try_shape_match(ms_name, ms_param, weights)
        if match_result.torch_data is not None:
            return WeightManager._try_load_parameter(
                ms_param,
                match_result.torch_data,
                match_result.match_type,
                match_result.torch_name,
            )

        return False

    @staticmethod
    def _report_loading_results(result: WeightLoadResult, strict: bool) -> None:
        """报告加载结果"""
        if result.missing_keys and strict:
            print(f"警告: {len(result.missing_keys)}/{result.total_count} 参数未加载")
            print(
                f"未加载的参数: {result.missing_keys[:5]}{'...' if len(result.missing_keys) > 5 else ''}"
            )

    @staticmethod
    def load_mindspore_weights(
        model, weights_path: str, strict: bool = True
    ) -> List[str]:
        """为MindSpore模型加载权重"""
        # 加载权重和映射关系
        weights = load_file(weights_path)
        torch_to_ms_map = WeightManager._get_torch_to_ms_map()
        WeightManager._try_load_metadata(weights_path)

        # 获取模型参数
        ms_params = list(model.get_parameters())
        missing_keys = []
        loaded_count = 0

        # 尝试加载每个参数
        for ms_param in ms_params:
            if WeightManager._match_parameter(ms_param, weights, torch_to_ms_map):
                loaded_count += 1
            else:
                missing_keys.append(ms_param.name)
                if strict:
                    print(f"缺失参数: {ms_param.name}, 形状: {ms_param.shape}")

        # 报告加载结果
        result = WeightLoadResult(missing_keys, loaded_count, len(ms_params))
        WeightManager._report_loading_results(result, strict)

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
