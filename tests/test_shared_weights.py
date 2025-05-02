import os
import sys
import click
import numpy as np
import json

# 确保能导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zwx_ms.utils.weight_utils import WeightManager
from zwx_ms.model.llama_pt import create_test_model as create_torch_model
from zwx_ms.model.llama_ms import create_test_model as create_ms_model
from zwx_ms.model.llama_ms import load_model_with_weights


def get_model_config(small=True):
    """获取模型配置"""
    if small:
        return {
            "vocab_size": 32000,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_hidden_layers": 2,
            "num_attention_heads": 12,
            "layer_norm_eps": 1e-5,
        }
    else:
        return {
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "layer_norm_eps": 1e-5,
        }

def generate_weights(output_path, small=True):
    """生成统一权重文件"""
    print(f"正在生成统一权重文件: {output_path}")
    
    # 获取模型配置
    config = get_model_config(small)
    
    # 生成权重文件
    WeightManager.generate_shared_weights(config, output_path, seed=42)
    
    print(f"权重文件生成完成: {output_path}")

def test_pytorch_loading(weights_path):
    """测试PyTorch模型加载权重"""
    try:
        import torch
        from zwx_ms.model.llama_pt import create_test_model as create_torch_model
    except ImportError:
        print("未安装PyTorch或导入错误，跳过PyTorch测试")
        return False
    
    print("\n=== 测试PyTorch模型权重加载 ===")
    
    # 创建模型
    model = create_torch_model()
    
    # 加载权重
    missing_keys = model.load_weights(weights_path)
    
    # 测试模型
    inputs = torch.randint(0, 32000, (1, 10))
    with torch.no_grad():
        outputs = model(inputs)
    
    print(f"PyTorch模型输出形状: {outputs.shape}")
    
    return not missing_keys

def test_mindspore_loading(weights_path):
    """测试MindSpore模型加载权重"""
    try:
        import mindspore as ms
        from zwx_ms.model.llama_ms import create_test_model as create_ms_model
        from zwx_ms.model.llama_ms import load_model_with_weights
    except ImportError:
        print("未安装MindSpore或导入错误，跳过MindSpore测试")
        return False
    
    print("\n=== 测试MindSpore模型权重加载 ===")
    
    # 设置MindSpore上下文
    ms.set_context(mode=ms.PYNATIVE_MODE)
    
    # 创建并加载模型
    model = load_model_with_weights(weights_path)
    
    # 测试模型
    inputs = ms.Tensor(np.random.randint(0, 32000, (1, 10)), ms.int32)
    outputs = model(inputs)
    
    print(f"MindSpore模型输出形状: {outputs.shape}")
    
    return True

def check_weights_compatibility(weights_path):
    """检查权重与两个框架的兼容性"""
    try:
        import torch
        import mindspore as ms
        from zwx_ms.model.llama_pt import create_test_model as create_torch_model
        from zwx_ms.model.llama_ms import create_test_model as create_ms_model
    except ImportError:
        print("导入错误，无法进行兼容性检查")
        return
    
    print("\n=== 检查权重兼容性 ===")
    
    # 创建模型
    torch_model = create_torch_model()
    
    # 设置MindSpore上下文
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms_model = create_ms_model()
    
    # 进行兼容性检查
    result = WeightManager.check_weights_compatibility(torch_model, ms_model, weights_path)
    
    # 打印结果
    print(f"权重文件中参数总数: {result['weights_total']}")
    print(f"PyTorch模型可加载参数比例: {result['torch_check']['loadable_percent']}")
    print(f"MindSpore模型可加载参数比例: {result['mindspore_check']['loadable_percent']}")
    
    if result['torch_check']['missing_params'] > 0:
        print(f"PyTorch模型缺失参数数量: {result['torch_check']['missing_params']}")
        if result['torch_check']['missing_details']:
            print(f"部分缺失参数: {', '.join(result['torch_check']['missing_details'][:5])}")
    
    if result['mindspore_check']['missing_params'] > 0:
        print(f"MindSpore模型缺失参数数量: {result['mindspore_check']['missing_params']}")
        if result['mindspore_check']['missing_details']:
            print(f"部分缺失参数: {', '.join(result['mindspore_check']['missing_details'][:5])}")
    
    # 保存报告到文件
    with open(weights_path.replace('.safetensors', '_compatibility.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"兼容性报告已保存到: {weights_path.replace('.safetensors', '_compatibility.json')}")

def test_model_outputs(weights_path):
    """测试加载相同权重的两个模型输出是否一致"""
    try:
        import torch
        import mindspore as ms
        import numpy as np
        from zwx_ms.model.llama_pt import create_test_model as create_torch_model
        from zwx_ms.model.llama_ms import create_test_model as create_ms_model
    except ImportError:
        print("导入错误，无法进行输出一致性测试")
        return
    
    print("\n=== 测试模型输出一致性 ===")
    
    # 创建种子以确保相同输入
    np.random.seed(42)
    
    # 创建输入数据
    input_ids_np = np.random.randint(0, 32000, (1, 10))
    
    # PyTorch模型
    torch_model = create_torch_model()
    torch_model.load_weights(weights_path)
    torch_input = torch.tensor(input_ids_np, dtype=torch.long)
    with torch.no_grad():
        torch_output = torch_model(torch_input).numpy()
    
    # MindSpore模型
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms_model = create_ms_model()
    ms_model.load_weights(weights_path)
    ms_input = ms.Tensor(input_ids_np, ms.int32)
    ms_output = ms_model(ms_input).asnumpy()
    
    # 比较输出
    diff = np.abs(torch_output - ms_output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"最大绝对差异: {max_diff:.2e}")
    print(f"平均绝对差异: {mean_diff:.2e}")
    print(f"输出形状: PyTorch {torch_output.shape} vs MindSpore {ms_output.shape}")
    
    if max_diff < 1e-3:
        print("测试通过! 两个模型输出基本一致")
    else:
        print("测试失败! 两个模型输出存在明显差异")
        
        # 找到差异最大的位置
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        torch_val = torch_output[max_idx]
        ms_val = ms_output[max_idx]
        print(f"最大差异位置: {max_idx}")
        print(f"PyTorch值: {torch_val:.6f}, MindSpore值: {ms_val:.6f}")

@click.command()
@click.option("--output-dir", type=str, default="weights", help="权重输出目录")
@click.option("--small/--large", default=True, help="使用小型模型配置")
@click.option("--skip-generate", is_flag=True, help="跳过权重生成")
@click.option("--check-compatibility", is_flag=True, help="检查权重兼容性")
@click.option("--test-outputs", is_flag=True, help="测试模型输出一致性")
def main(output_dir, small, skip_generate, check_compatibility, test_outputs):
    """生成并测试PyTorch和MindSpore共享权重"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 权重文件路径
    weights_path = os.path.join(output_dir, "llama_shared.safetensors")
    
    # 生成权重
    if not skip_generate or not os.path.exists(weights_path):
        generate_weights(weights_path, small)
    
    # 测试PyTorch加载
    pytorch_success = test_pytorch_loading(weights_path)
    
    # 测试MindSpore加载
    mindspore_success = test_mindspore_loading(weights_path)
    
    # 兼容性检查
    if check_compatibility:
        check_weights_compatibility(weights_path)
    
    # 测试模型输出一致性
    if test_outputs:
        test_model_outputs(weights_path)
    
    # 总结
    print("\n=== 测试总结 ===")
    print(f"权重文件: {weights_path}")
    print(f"PyTorch加载: {'成功' if pytorch_success else '失败'}")
    print(f"MindSpore加载: {'成功' if mindspore_success else '失败'}")
    print(f"权重文件大小: {os.path.getsize(weights_path) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    main() 