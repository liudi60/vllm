# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from typing import Any, Callable

import vllm.envs as envs

logger = logging.getLogger(__name__)

DEFAULT_PLUGINS_GROUP = 'vllm.general_plugins'

# make sure one process only loads plugins once
plugins_loaded = False


def load_plugins_by_group(group: str) -> dict[str, Callable[[], Any]]:
    import sys
    if sys.version_info < (3, 10):
        from importlib_metadata import entry_points
    else:
        from importlib.metadata import entry_points

    allowed_plugins = envs.VLLM_PLUGINS

    logger.warning(f'===== load_plugins_by_group, envs.VLLM_PLUGINS={envs.VLLM_PLUGINS}')  # None
    '''
    entry_points={
        "vllm.platform_plugins": ["ascend = vllm_ascend:register"],
        "vllm.general_plugins":
        ["ascend_enhanced_model = vllm_ascend:register_model"],
    }
    '''

    '''
    1.插件定义位置（注：使用setup.py方式定义的entry_points插件）
    group=vllm.platform_plugins 这个插件（或者叫entry_points组）是在该文件中注册的：/home/liudi/vllm-workspace/vllm-ascend/setup.py
    
    2.安装插件
    在 /home/liudi/vllm-workspace/vllm-ascend 目录下执行：pip install -e . ，即可完成安装 vllm.platform_plugins 这个插件。
    
    3.插件使用
    discovered_plugins = entry_points(group=group)  # 查询所有已安装的python包（每个包是一个dist对象）包含的entry_points，然后筛选出来对应group名的entry_points 
    
    4.插件定义代码：
    setup.py中注册相关代码为：
        ...
        setup(
            name="vllm_ascend",
            # Follow:
            # https://packaging.python.org/en/latest/specifications/version-specifiers
            version=VERSION,
            author="vLLM-Ascend team",
            license="Apache 2.0",
            description="vLLM Ascend backend plugin",
            long_description=read_readme(),
            long_description_content_type="text/markdown",
            url="https://github.com/vllm-project/vllm-ascend",
            project_urls={
                "Homepage": "https://github.com/vllm-project/vllm-ascend",
            },
            # TODO: Add 3.12 back when torch-npu support 3.12
            classifiers=[
                "Programming Language :: Python :: 3.9",
                "Programming Language :: Python :: 3.10",
                "Programming Language :: Python :: 3.11",
                "License :: OSI Approved :: Apache Software License",
                "Intended Audience :: Developers",
                "Intended Audience :: Information Technology",
                "Intended Audience :: Science/Research",
                "Topic :: Scientific/Engineering :: Artificial Intelligence",
                "Topic :: Scientific/Engineering :: Information Analysis",
            ],
            packages=find_packages(exclude=("docs", "examples", "tests*", "csrc")),
            python_requires=">=3.9",
            install_requires=get_requirements(),
            ext_modules=ext_modules,
            cmdclass=cmdclass,
            extras_require={},
            entry_points={
                "vllm.platform_plugins": ["ascend = vllm_ascend:register"],
                "vllm.general_plugins":
                ["ascend_enhanced_model = vllm_ascend:register_model"],
            },
        )
    '''
    logger.warning(f'===== load_plugins_by_group, group={group}')  # group=vllm.platform_plugins
    discovered_plugins = entry_points(group=group)  # 这里边查找到vllm-ascend插件的。group："vllm.platform_plugins"
    '''
    如下打印：
    discovered_plugins=[EntryPoint(name='ascend', value='vllm_ascend:register', group='vllm.platform_plugins')]
    
    value='vllm_ascend:register' 指的是 /home/liudi/vllm-workspace/vllm-ascend/vllm_ascend/__init__.py 中 register()函数
    
    register()函数如下：
        def register():
            """Register the NPU platform."""
            return "vllm_ascend.platform.NPUPlatform"
    '''
    logger.warning(f'===== load_plugins_by_group, discovered_plugins={discovered_plugins}')
    if len(discovered_plugins) == 0:
        logger.debug("No plugins for group %s found.", group)
        return {}

    # Check if the only discovered plugin is the default one
    is_default_group = (group == DEFAULT_PLUGINS_GROUP)
    # Use INFO for non-default groups and DEBUG for the default group
    log_level = logger.debug if is_default_group else logger.info

    log_level("Available plugins for group %s:", group)
    for plugin in discovered_plugins:
        log_level("- %s -> %s", plugin.name, plugin.value)

    if allowed_plugins is None:
        log_level("All plugins in this group will be loaded. "
                  "Set `VLLM_PLUGINS` to control which plugins to load.")

    plugins = dict[str, Callable[[], Any]]()
    for plugin in discovered_plugins:
        if allowed_plugins is None or plugin.name in allowed_plugins:
            if allowed_plugins is not None:
                log_level("Loading plugin %s", plugin.name)

            try:
                func = plugin.load()  # todo 这里的func就是 /home/liudi/vllm-workspace/vllm-ascend/vllm_ascend/__init__.py 中 register()函数
                plugins[plugin.name] = func
            except Exception:
                logger.exception("Failed to load plugin %s", plugin.name)

    return plugins


def load_general_plugins():
    """WARNING: plugins can be loaded for multiple times in different
    processes. They should be designed in a way that they can be loaded
    multiple times without causing issues.
    """
    global plugins_loaded
    if plugins_loaded:
        return
    plugins_loaded = True

    '''
    在 /vllm-ascend/setup.py 中注册了两个entry_points组，代码如下：
        entry_points={
            "vllm.platform_plugins": ["ascend = vllm_ascend:register"],
            "vllm.general_plugins":
            ["ascend_enhanced_model = vllm_ascend:register_model"],
        }
    
    DEFAULT_PLUGINS_GROUP = 'vllm.general_plugins'
    
    func() 是 vllm_ascend 仓的 vllm_ascend/__init__.py 中 register_model() 函数 
    
    '''
    plugins = load_plugins_by_group(group=DEFAULT_PLUGINS_GROUP)  # DEFAULT_PLUGINS_GROUP = 'vllm.general_plugins'
    # ===== load_general_plugins, plugins={'lora_filesystem_resolver': <function register_filesystem_resolver at 0xfffd06f0d760>, 'ascend_enhanced_model': <function register_model at 0xfffdb3d7e340>}
    logger.warning(f'===== load_general_plugins, plugins={plugins}')
    # general plugins, we only need to execute the loaded functions
    for func in plugins.values():
        func()
