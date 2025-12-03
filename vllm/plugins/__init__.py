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
    logger.warning(f'===== load_plugins_by_group, group={group}')  # group=vllm.platform_plugins
    discovered_plugins = entry_points(group=group)  # 这里边查找到vllm-ascend插件的。group："vllm.platform_plugins"
    # discovered_plugins=[EntryPoint(name='ascend', value='vllm_ascend:register', group='vllm.platform_plugins')]
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
                func = plugin.load()
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

    plugins = load_plugins_by_group(group=DEFAULT_PLUGINS_GROUP)
    logger.warning(f'===== load_general_plugins, plugins={plugins}')
    # general plugins, we only need to execute the loaded functions
    for func in plugins.values():
        func()
