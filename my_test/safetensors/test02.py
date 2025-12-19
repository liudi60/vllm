# 🔍 文件结构（.safetensors 内部）
# 一个 .safetensors 文件由两部分组成：
#
# Header（JSON）：描述每个张量的名称、形状、数据类型、字节偏移。
# Data（Raw bytes）：连续存储所有张量的原始二进制数据。
# 例如：
#
# json
# 编辑
# {
#   "weight": {"dtype": "F32", "shape": [5,10], "data_offsets": [0, 200]},
#   "bias":   {"dtype": "F32", "shape": [5],    "data_offsets": [200, 220]}
# }
# 💡 这种设计使得加载时可以 内存映射（mmap），无需将整个文件读入内存。
#
# ⚡ 高级用法：内存映射加载（超大模型）
# 对于超大模型（如 70B LLM），可使用 safe_open 实现 按需加载张量，节省内存：
# ✅ 适用于 vLLM、llama.cpp、text-generation-inference 等推理引擎。


from safetensors import safe_open

# 不一次性加载所有张量
with safe_open("/home/ml/weight/Qwen3-8B-W8A8/quant_model_weight_w8a8.safetensors", framework="pt", device="cpu") as f:
    '''
    这些keys就是config.json中定义的。
    lm_head.weight
    model.embed_tokens.weight
    model.layers.0.input_layernorm.weight
    model.layers.0.mlp.down_proj.weight
    model.layers.0.mlp.gate_proj.deq_scale
    model.layers.0.mlp.gate_proj.input_offset
    model.layers.0.mlp.gate_proj.input_scale
    model.layers.0.mlp.gate_proj.quant_bias
    model.layers.0.mlp.gate_proj.weight
    model.layers.0.mlp.gate_proj.weight_offset
    model.layers.0.mlp.gate_proj.weight_scale
    model.layers.0.mlp.up_proj.deq_scale
    model.layers.0.mlp.up_proj.input_offset
    model.layers.0.mlp.up_proj.input_scale
    model.layers.0.mlp.up_proj.quant_bias
    model.layers.0.mlp.up_proj.weight
    model.layers.0.mlp.up_proj.weight_offset
    model.layers.0.mlp.up_proj.weight_scale
    model.layers.0.post_attention_layernorm.weight
    model.layers.0.self_attn.k_norm.weight
    model.layers.0.self_attn.k_proj.deq_scale
    model.layers.0.self_attn.k_proj.input_offset
    model.layers.0.self_attn.k_proj.input_scale
    model.layers.0.self_attn.k_proj.quant_bias
    model.layers.0.self_attn.k_proj.weight
    model.layers.0.self_attn.k_proj.weight_offset
    model.layers.0.self_attn.k_proj.weight_scale
    model.layers.0.self_attn.o_proj.deq_scale
    model.layers.0.self_attn.o_proj.input_offset
    model.layers.0.self_attn.o_proj.input_scale
    model.layers.0.self_attn.o_proj.quant_bias
    '''
    keys = f.keys()
    i = 0
    for key in keys:
        if ++i == 20:
            break
        print(key)
    print()
    # 只加载你需要的张量
    weight = f.get_tensor("model.layers.0.self_attn.q_proj.weight")
    print(weight.shape)  # torch.Size([4096, 4096])

    print(f'state_dict={f}')
