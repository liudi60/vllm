safetensors 是由 Hugging Face 开发的一种 安全、快速、无 pickle 的张量序列化格式，主要用于存储和加载深度学习模型权重（如 PyTorch、TensorFlow 模型），避免了传统 .bin 或 pytorch_model.bin 使用 pickle 带来的安全隐患（如任意代码执行）。

✅ 为什么使用 safetensors？

问题 传统 PyTorch .bin safetensors
------ --------------------- ---------------
安全性 使用 pickle，可执行任意代码（危险！） 纯二进制 + 元数据 JSON，无法执行代码
加载速度 较慢（需反序列化 Python 对象） 极快（直接内存映射，zero-copy）
跨语言支持 仅限 Python 支持 Python、Rust、C++、JavaScript 等
文件大小 相同 相同（只存张量数据）
🔒 安全提示：从不可信来源下载的 .bin 模型可能包含恶意代码，而 .safetensors 文件 绝对安全。

📦 安装

bash
pip install safetensors

🧪 基本使用（PyTorch）
1. 保存模型权重为 .safetensors

python
import torch
from safetensors.torch import save_file
假设你有一个 state_dict
model = torch.nn.Linear(10, 5)
state_dict = model.state_dict()
保存（必须是 dict[str, torch.Tensor]）
save_file(state_dict, "model.safetensors")
⚠️ 注意：state_dict 中的值必须是 torch.Tensor，不能是 int、str 等（会报错）。

2. 从 .safetensors 加载权重

python
from safetensors.torch import load_file
加载为 dict[str, torch.Tensor]
loaded_state_dict = load_file("model.safetensors")
加载到模型
model = torch.nn.Linear(10, 5)
model.load_state_dict(loaded_state_dict)

3. 与 Hugging Face Transformers 集成

Hugging Face 已全面支持 safetensors。你只需：
✅ 下载时自动优先使用 .safetensors
python
from transformers import AutoModel
如果 hub 上有 .safetensors 文件，会自动下载并加载（更安全更快）
model = AutoModel.from_pretrained("bert-base-uncased")
✅ 上传模型时生成 .safetensors
python
from transformers import AutoModel
from huggingface_hub import snapshot_download

model = AutoModel.from_pretrained("your-model")
model.save_pretrained("my-model") # 默认仍保存 .bin
要生成 .safetensors，需额外安装并转换：
!pip install -U huggingface_hub
!huggingface-cli convert --model your-model-name --output my-model-safetensors

或者在代码中转换：
python
from transformers import AutoModel
from safetensors.torch import save_file

model = AutoModel.from_pretrained("bert-base-uncased")
save_file(model.state_dict(), "model.safetensors")

🔍 文件结构（.safetensors 内部）

一个 .safetensors 文件由两部分组成：

1. Header（JSON）：描述每个张量的名称、形状、数据类型、字节偏移。
2. Data（Raw bytes）：连续存储所有张量的原始二进制数据。

例如：
json
{
"weight": {"dtype": "F32", "shape": [5,10], "data_offsets": [0, 200]},
"bias": {"dtype": "F32", "shape": [5], "data_offsets": [200, 220]}
}
💡 这种设计使得加载时可以 内存映射（mmap），无需将整个文件读入内存。

⚡ 高级用法：内存映射加载（超大模型）

对于超大模型（如 70B LLM），可使用 safe_open 实现 按需加载张量，节省内存：

python
from safetensors import safe_open
不一次性加载所有张量
with safe_open("model.safetensors", framework="pt", device="cpu") as f:
# 只加载你需要的张量
weight = f.get_tensor("transformer.h.0.attn.c_attn.weight")
print(weight.shape)
✅ 适用于 vLLM、llama.cpp、text-generation-inference 等推理引擎。

🌐 其他框架支持

框架 加载方式
------ --------
TensorFlow safetensors.tensorflow.load_file
Flax/JAX safetensors.flax.load_file
NumPy safetensors.numpy.load_file

❌ 常见错误

1. 尝试保存非 Tensor 数据
python
save_file({"version": "1.0"}, "meta.safetensors") # ❌ 报错！

→ 解决：只保存 torch.Tensor。

2. 文件名不带 .safetensors
→ 虽然不是强制，但建议使用标准后缀。

3. 在不支持的设备上加载
python
load_file("model.safetensors", device="cuda") # 需确保 CUDA 可用

✅ 总结：何时使用 safetensors？

场景 推荐
------ ------
从 Hugging Face Hub 下载模型 ✅ 优先使用 .safetensors
分享模型给他人 ✅ 用 .safetensors 更安全
部署生产环境 ✅ 避免 pickle 风险
加载超大模型 ✅ 用 safe_open 节省内存
保存训练中间 checkpoint ⚠️ 可选（.bin 也可，但 .safetensors 更快）
📌 最佳实践：
“永远不要运行来自未知来源的 .bin 模型；优先选择 .safetensors。”

官方仓库：https://github.com/huggingface/safetensors
文档：https://huggingface.co/docs/safetensors/index