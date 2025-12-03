from importlib.metadata import entry_points


# print(f'console_scripts:')
# # console_scripts
# renderers = entry_points().select(group="console_scripts")
# for ep in renderers:
#     func = ep.load()
#     print(f'===== func={func}')
#     # func()


print(f'myapp_renderers:')
# 发现所有渲染器插件
renderers = entry_points().select(group="myapp_renderers")
for ep in renderers:
    RendererClass = ep.load()
    renderer = RendererClass()
    print(f"Loaded: {ep.name}")
    

print(f'vllm_workers:')
renderers = entry_points().select(group="vllm_workers")
for ep in renderers:
    RendererClass = ep.load()
    renderer = RendererClass()
    print(f"Loaded: {ep.name}")


