# PyTorch和MindSpore权重共享工具

本工具提供了PyTorch和MindSpore深度学习框架之间共享模型权重的功能，解决了在两个框架中使用相同模型参数的需求，方便进行精度比较和调试。

## 功能特点

1. 使用safetensors格式保存统一的权重文件，保证安全和跨平台兼容性
2. 为PyTorch和MindSpore提供了统一的权重加载接口
3. 支持权重兼容性检查，确保两个框架能够正确加载参数
4. 提供权重加载验证，确保模型各层参数被正确初始化
5. 支持输出一致性测试，验证两个框架使用相同权重后输出是否一致

## 安装依赖

```bash
pip install torch mindspore safetensors numpy
```

## 使用方法

### 1. 生成和测试共享权重

运行`test_shared_weights.py`脚本可以生成统一的权重文件，并测试PyTorch和MindSpore模型能够正确加载这些权重：

```bash
python test_shared_weights.py --output-dir weights --test-outputs --check-compatibility
```

参数说明：
- `--output-dir`: 指定权重文件输出目录
- `--small`: 使用小型模型配置（默认）
- `--skip-generate`: 如果权重文件已存在，可以跳过生成步骤
- `--check-compatibility`: 检查权重与两个框架模型的兼容性
- `--test-outputs`: 测试两个框架模型加载相同权重后输出是否一致

### 2. 在PyTorch模型中加载权重

```python
from model.torch.llama import create_test_model

# 创建模型
model = create_test_model()

# 加载权重
weights_path = "weights/llama_shared.safetensors"
model.load_weights(weights_path)

# 使用模型
import torch
inputs = torch.randint(0, 32000, (1, 10))
with torch.no_grad():
    outputs = model(inputs)
```

### 3. 在MindSpore模型中加载权重

```python
import mindspore as ms
from model.mindspore.llama import load_model_with_weights

# 设置MindSpore上下文
ms.set_context(mode=ms.PYNATIVE_MODE)

# 创建并加载模型
weights_path = "weights/llama_shared.safetensors"
model = load_model_with_weights(weights_path)

# 使用模型
import numpy as np
inputs = ms.Tensor(np.random.randint(0, 32000, (1, 10)), ms.int32)
outputs = model(inputs)
```

### 4. 使用WeightManager生成自定义模型权重

如果需要为自定义模型生成权重，可以使用`WeightManager`类：

```python
from weight_utils import WeightManager

# 模型配置
model_config = {
    "vocab_size": 32000,
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_hidden_layers": 2,
    "num_attention_heads": 12,
    "layer_norm_eps": 1e-5,
}

# 生成权重文件
weights_path = "weights/custom_model.safetensors"
WeightManager.generate_shared_weights(model_config, weights_path, seed=42)
```

## 原理说明

### 权重共享实现原理

1. **统一格式**：使用safetensors格式保存权重，该格式支持高效读取和序列化，并且比pickle更安全
2. **命名规范**：采用统一的参数命名规范，确保两个框架可以正确映射参数
3. **元数据存储**：在权重文件中包含元数据，记录参数映射关系和模型配置
4. **智能匹配**：在加载过程中，会尝试智能匹配参数名称，处理框架间的命名差异

### 如何保证权重正确加载

1. **形状检查**：在加载时验证参数形状是否匹配
2. **名称映射**：提供灵活的参数名称映射机制，解决两个框架命名不一致的问题
3. **加载验证**：通过检查未加载的参数数量，确保模型的完整性
4. **输出一致性**：提供测试功能，验证两个框架使用相同权重和输入时的输出一致性

## 文件结构

- `weight_utils.py`: 核心权重管理工具类，提供权重生成和加载功能
- `test_shared_weights.py`: 权重生成和测试脚本
- `model/torch/llama.py`: PyTorch版本的LLaMA模型实现
- `model/mindspore/llama.py`: MindSpore版本的LLaMA模型实现

## 注意事项

1. 确保PyTorch和MindSpore模型结构保持一致，包括层数、隐藏大小等配置
2. 参数名称映射机制虽然灵活，但仍需保持基本命名规则的一致性
3. 可能存在浮点精度差异，导致两个框架的输出略有不同，这是正常现象
4. 权重文件较大时，加载可能需要更长时间，请耐心等待

## 自定义扩展

要支持新的模型结构，需要：

1. 在`weight_utils.py`中扩展`WeightManager.generate_shared_weights`方法，添加新模型的参数生成逻辑
2. 在新模型类中实现`load_weights`方法，调用`WeightManager.load_torch_weights`或`WeightManager.load_mindspore_weights`
3. 更新参数名称映射规则，确保新添加的参数能被正确匹配

## 可能的错误及解决方法

1. **找不到模块**：确保已安装所有依赖，并且目录结构正确
2. **参数形状不匹配**：检查两个框架的模型结构是否一致
3. **权重加载失败**：检查参数命名是否遵循规范，可能需要更新映射规则
4. **输出差异过大**：检查模型实现中是否有算法差异，例如不同的初始化或激活函数

# PyTorch 和 MindSpore 框架间共享权重实现

## 问题描述

在测试精度比较工具时出现了 KeyError 错误，原因是 PyTorch 模型使用 "n_heads" 键而 MindSpore 使用 "num_attention_heads" 键。这导致在共享权重过程中出现了不一致。

## 解决方案

1. 修复了 PyTorch 模型中 LlamaAttention 类的初始化，将 "n_heads" 改为 "num_attention_heads"，使其与 MindSpore 模型保持一致。

2. 确认 WeightManager 类中的 generate_shared_weights 和 load_mindspore_weights 方法已经可以正常工作。

3. main.py 中添加了以下命令行选项：
   - `--generate_weights`: 生成共享权重文件
   - `--test_weights`: 测试权重加载功能
   - `--test_precision`: 运行精度测试

4. test_precision.py 中增加了权重生成和加载功能，确保两个框架使用相同的初始权重。

## 使用方法

1. 生成共享权重文件：
```bash
python main.py --generate_weights
```

2. 测试权重加载功能：
```bash
python main.py --test_weights
```

3. 运行精度测试：
```bash
python main.py --test_precision
```

4. 生成权重并进行测试：
```bash
python main.py --generate_weights --test_weights --test_precision
```

## 实现细节

### 1. 权重生成 (WeightManager.generate_shared_weights)

权重生成方法会创建一个包含模型所有参数的字典，并保存为 safetensors 格式。生成的权重文件包含元数据，用于不同框架间的参数映射。

### 2. PyTorch 权重加载 (WeightManager.load_torch_weights)

加载方法会尝试将 safetensors 文件中的权重加载到 PyTorch 模型中。如果参数名称不完全匹配，会尝试进行映射，确保所有参数都能正确加载。

### 3. MindSpore 权重加载 (WeightManager.load_mindspore_weights)

类似地，MindSpore 权重加载方法会尝试将 safetensors 文件中的权重加载到 MindSpore 模型中，需要适应 MindSpore 的特定数据结构和参数管理方式。

### 4. 兼容性检查 (WeightManager.check_weights_compatibility)

此方法可以检查权重文件与两个框架模型的兼容性，提供详细的兼容性报告，包括可加载参数比例、缺失参数和形状不匹配参数等信息。

## 关键改进

1. 统一了 PyTorch 和 MindSpore 模型中的参数命名约定，使两个框架可以共享相同的权重文件。

2. 添加了自动生成权重和加载权重的功能，简化了测试流程。

3. 改进了精度测试程序，确保测试前两个框架的模型使用相同的初始权重。

通过这些改动，我们成功解决了两个框架间共享权重的问题，提高了精度测试的准确性和可靠性。 