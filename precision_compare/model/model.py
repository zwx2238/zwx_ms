from precision_compare.model.config import get_llama_config


def create_test_model(model_class):
    """创建测试模型"""
    config = get_llama_config(small=True)
    model = model_class(config)
    return model
