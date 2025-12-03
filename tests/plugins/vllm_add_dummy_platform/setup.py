# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from setuptools import setup

# 没走这里，不是在这注册的
print(f'===== setup vllm_add_dummy_platform, vllm.platform_plugins')

setup(
    name='vllm_add_dummy_platform',
    version='0.1',
    packages=['vllm_add_dummy_platform'],
    entry_points={
        'vllm.platform_plugins': [
            "dummy_platform_plugin = vllm_add_dummy_platform:dummy_platform_plugin"  # noqa
        ],
        "vllm.general_plugins":
        ["dummy_custom_ops = vllm_add_dummy_platform:register_ops"],
    })
