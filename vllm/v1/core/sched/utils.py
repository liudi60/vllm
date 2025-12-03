# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
from typing import Optional

import torch

from vllm.v1.request import Request, RequestStatus


def remove_all(lst: list, items_to_remove: set) -> list:
    """Remove all items from a list that are in the items_to_remove set.

    This method optimizes for the common case of removing a single item,
    falling back to list comprehension for multiple items.

    Args:
        lst: The list to remove items from
        items_to_remove: Set of items to remove

    Returns:
        Either the modified original list (for single item removal) or
        a new list (for multiple item removal). Callers should use the
        returned value.

    Note:
        For single item removal, this modifies the original list in-place
        and returns it. For multiple items, it creates and returns a new list.
    """
    if not items_to_remove:
        return lst

    if len(items_to_remove) == 1:
        # Fast path for single item removal (most common case)
        item = next(iter(items_to_remove))
        with contextlib.suppress(ValueError):
            lst.remove(item)
        return lst
    # For multiple items, use list comprehension
    return [item for item in lst if item not in items_to_remove]


def check_stop(request: Request,
               max_model_len: int,
               pooler_output: Optional[torch.Tensor] = None) -> bool:
    if (request.num_tokens >= max_model_len
            or request.num_output_tokens >= request.max_tokens):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True

    '''
    ✅ 1. request.pooling_params 是什么？
    这是一个可选字段，表示该请求不是用于生成文本，而是用于获取句子/文本的 embedding 向量。
    它对应 vLLM 中的 PoolingParams 类型（或类似结构），包含：
    pooling_type: 如 mean, cls, last 等
    是否需要归一化等参数
    💡 这类请求常见于 embedding 模型（如 BAAI/bge-small-en、sentence-transformers 系列），它们没有“生成 token”的过程，而是在 prefill 阶段结束后直接通过 pooler 层输出一个向量。
    
    ✅ 2. pooler_output is not None 是什么意思？
    在模型执行 execute_model 后，如果模型支持 pooling（如 transformers 中的 BertModel、RobertaModel 带 pooler 层），vLLM 会尝试提取 pooler_output。
    对于 纯 decoder 模型（如 Llama），通常没有 pooler_output，所以为 None。
    但对于 encoder-only 或 encoder-decoder 模型用于 embedding 任务，pooler_output 会在 prefill 完成后立即生成。
    ✅ 所以：pooler_output is not None 表示 模型已经成功计算出 embedding 向量。
    
    ✅ 3. 为什么此时标记为 FINISHED_STOPPED？
    对于 pooling 请求：
    不需要 decode 生成新 token
    prefill 完成就等于整个推理完成
    因此，一旦拿到 pooler_output，就应立即将请求状态设为 已完成（FINISHED_STOPPED），并返回结果。
    📝 注意：虽然叫 STOPPED，但这里并不是因为遇到 stop token，而是“任务自然结束”。vLLM 复用了这个状态枚举。
    '''
    if request.pooling_params:
        if pooler_output is not None:
            request.status = RequestStatus.FINISHED_STOPPED
            return True
        return False

    sampling_params = request.sampling_params
    assert sampling_params is not None
    last_token_id = request.output_token_ids[-1]
    if (not sampling_params.ignore_eos
            and last_token_id == request.eos_token_id):
        request.status = RequestStatus.FINISHED_STOPPED
        return True

    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = last_token_id
        return True
    return False
