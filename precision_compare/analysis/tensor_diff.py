from dataclasses import dataclass
from typing import Tuple


@dataclass
class TensorDiff:
    # TODO 调整顺序，减少调用位置
    module_name: str
    tensor_type: str  # 'inputs', 'outputs', 或 'parameters'
    max_abs_diff: float
    cosine_similarity: float  # 添加余弦相似度
    location: Tuple[int, ...]  # 最大差异的位置
    shape: Tuple[int, ...]  # tensor的形状
    max_rel_diff: float  # 最大差异位置处的相对误差
    execution_index: int = 0
