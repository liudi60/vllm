# safetensors 基本使用（PyTorch）

# 1. 保存模型权重为 .safetensors
import torch
from safetensors.torch import save_file

# 假设你有一个 state_dict
model = torch.nn.Linear(10, 5)
state_dict = model.state_dict()

print(f'state_dict={state_dict}')

# 保存（必须是 dict[str, torch.Tensor]）
save_file(state_dict, "my_model.safetensors")

print(model.weight.data)



# 2. 从 .safetensors 加载权重
from safetensors.torch import load_file

# 加载为 dict[str, torch.Tensor]
loaded_state_dict = load_file("my_model.safetensors")

# 加载到模型
model = torch.nn.Linear(10, 5)
model.load_state_dict(loaded_state_dict)

print(model.weight.data)
