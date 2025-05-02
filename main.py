import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="测试深度学习框架的模型中间结果记录和分析工具")
    parser.add_argument("--framework", type=str, choices=["torch", "mindspore", "both"], default="both",
                       help="选择要测试的深度学习框架")
    parser.add_argument("--output_dir", type=str, default="dump_logs",
                       help="中间结果保存目录")
    parser.add_argument("--analyze", action="store_true",
                       help="分析对比两个框架的结果")
    parser.add_argument("--generate_weights", action="store_true",
                       help="生成共享权重文件")
    parser.add_argument("--force_regenerate", action="store_true",
                       help="强制重新生成权重文件（即使已存在）")
    parser.add_argument("--weights_dir", type=str, default="weights",
                       help="共享权重保存目录")
    parser.add_argument("--test_weights", action="store_true",
                       help="测试权重加载功能")
    parser.add_argument("--test_precision", action="store_true",
                       help="运行精度测试")
    parser.add_argument("--error_type", type=str, default="none",
                       choices=["none", "weight_noise", "dtype_cast", "activation_quantization"],
                       help="指定错误注入类型")
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.weights_dir, exist_ok=True)
    
    # 生成共享权重
    if args.generate_weights:
        try:
            print("=== 开始生成共享权重文件 ===")
            from weight_utils import WeightManager
            from model.torch.llama import get_llama_config
            
            # 创建权重目录
            os.makedirs(args.weights_dir, exist_ok=True)
            weights_path = os.path.join(args.weights_dir, "shared_weights.safetensors")
            
            # 检查是否已存在
            if os.path.exists(weights_path) and not args.force_regenerate:
                print(f"权重文件已存在: {weights_path}")
                print(f"如需重新生成，请添加 --force_regenerate 参数")
            else:
                config = get_llama_config(small=True)
                
                print(f"正在生成权重文件: {weights_path}")
                print(f"模型配置: vocab_size={config['vocab_size']}, hidden_size={config['hidden_size']}, num_attention_heads={config['num_attention_heads']}")
                
                WeightManager.generate_shared_weights(config, weights_path, seed=42)
                print(f"权重文件已生成: {weights_path}")
                
                # 检查文件是否真的存在
                if os.path.exists(weights_path):
                    print(f"成功创建权重文件，大小: {os.path.getsize(weights_path) / (1024 * 1024):.2f} MB")
                else:
                    print(f"警告: 权重文件生成过程完成，但未能找到文件: {weights_path}")
        except ImportError as e:
            print(f"导入错误: {str(e)}")
            print("可能未安装safetensors或其他依赖，请确保已安装所有必要包:")
            print("  pip install torch mindspore safetensors numpy")
        except Exception as e:
            print(f"生成权重文件失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 测试权重加载
    if args.test_weights:
        weights_path = os.path.join(args.weights_dir, "shared_weights.safetensors")
        if not os.path.exists(weights_path):
            print(f"权重文件不存在: {weights_path}")
            print("请先使用 --generate_weights 生成权重文件")
            return
        
        # 测试PyTorch
        if args.framework in ["torch", "both"]:
            try:
                print("\n=== 测试PyTorch权重加载 ===")
                from model.torch.llama import create_test_model
                
                model = create_test_model()
                model.load_weights(weights_path)
                print("PyTorch权重加载成功!")
            except ImportError:
                print("未安装PyTorch，跳过测试")
            except Exception as e:
                print(f"PyTorch权重加载失败: {str(e)}")
        
        # 测试MindSpore
        if args.framework in ["mindspore", "both"]:
            try:
                print("\n=== 测试MindSpore权重加载 ===")
                import mindspore as ms
                from model.mindspore.llama import load_model_with_weights
                
                ms.set_context(mode=ms.PYNATIVE_MODE)
                model = load_model_with_weights(weights_path)
                print("MindSpore权重加载成功!")
            except ImportError:
                print("未安装MindSpore，跳过测试")
            except Exception as e:
                print(f"MindSpore权重加载失败: {str(e)}")
    
    # 运行精度测试
    if args.test_precision:
        try:
            print("\n=== 运行精度测试 ===")
            from test_precision import run_test_case
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            test_dir = f"test_case_{args.error_type}_{timestamp}"
            
            test_config = {
                "error_type": args.error_type,
                "module_path": "model.layers.0",
                "framework": args.framework,
                "scale": 0.01,  # 权重噪声比例
                "dtype": "float16",  # 数据类型转换目标
                "bits": 8,  # 激活量化比特数
            }
            
            run_test_case(test_dir, test_config)
            print(f"精度测试完成，结果保存在: {test_dir}")
        except ImportError:
            print("未找到test_precision模块，无法运行精度测试")
        except Exception as e:
            print(f"精度测试失败: {str(e)}")
    
    # 测试PyTorch模型
    if not (args.generate_weights or args.test_weights or args.test_precision) and args.framework in ["torch", "both"]:
        try:
            print("=== 开始测试PyTorch模型 ===")
            from model.torch.llama import test_model_dump
            test_model_dump()
            print("PyTorch模型测试完成")
        except ImportError:
            print("未安装PyTorch或导入错误，跳过PyTorch测试")
        except Exception as e:
            print(f"PyTorch模型测试失败: {str(e)}")
    
    # 测试MindSpore模型
    if not (args.generate_weights or args.test_weights or args.test_precision) and args.framework in ["mindspore", "both"]:
        try:
            print("\n=== 开始测试MindSpore模型 ===")
            from model.mindspore.llama import test_model_dump
            test_model_dump()
            print("MindSpore模型测试完成")
        except ImportError:
            print("未安装MindSpore或导入错误，跳过MindSpore测试")
        except Exception as e:
            print(f"MindSpore模型测试失败: {str(e)}")
    
    # 分析比较结果
    if args.analyze:
        try:
            print("\n=== 开始分析比较两个框架的结果 ===")
            from analyzer import analyze_runs
            analyze_runs("torch_dump_logs", "mindspore_dump_logs", f"{args.output_dir}/comparison")
            print(f"分析结果已保存到 {args.output_dir}/comparison")
        except ImportError:
            print("未找到分析器模块，跳过结果分析")
        except Exception as e:
            print(f"分析过程出错: {str(e)}")

if __name__ == "__main__":
    main()

