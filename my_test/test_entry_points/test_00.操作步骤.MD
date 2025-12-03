# 1、项目目录为 my_project，安装命令：
pip install .

# 2、安装完后，就会把pyproject.toml中定义的两个entry_point group名为：myapp.renderers 和 vllm.worker注册完成（这两个是自定义group或者叫插件）。
# 其他 Python 程序可以通过 entry_points 发现自定义的插件，验证group是否注册ok，获取entry_point group代码如下：

from importlib.metadata import entry_points

# 发现所有渲染器插件
renderers = entry_points().select(group="myapp.renderers")
for ep in renderers:
    RendererClass = ep.load()
    renderer = RendererClass()
    print(f"Loaded: {ep.name}")

# 直接在终端上执行：mycli




