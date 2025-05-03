def get_llama_config(small=True):
    """获取Llama模型配置"""
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
