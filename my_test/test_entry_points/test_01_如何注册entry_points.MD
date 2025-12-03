在 Python 中，注册 entry_points 是通过项目的打包配置文件（如 pyproject.toml 或 setup.py）完成的。entry_points 是一种标准机制，用于声明插件、命令行工具或可被其他程序动态发现的功能模块。

✅ 一、什么是 entry_points？

entry_points 允许你在安装一个包时，向系统注册一些“命名入口点”，其他程序可以在运行时通过 importlib.metadata.entry_points() 动态发现并加载它们。

常见用途：
注册命令行工具（如 vllm, black, pytest）
实现插件系统（如数据库驱动、设备后端、格式化器等）

✅ 二、如何注册？两种方式
方式 1：使用 pyproject.toml（推荐，现代标准）

适用于使用 setuptools、hatch、poetry（需适配）等构建后端。

toml
pyproject.toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "my-package"
version = "0.1.0"
... 其他元数据
注册 entry_points
[project.entry-points]
1. 命令行脚本（console_scripts）
console_scripts = [
"mycli = mypackage.cli:main",
"mytool = mypackage.tools:run",
]
2. 自定义插件组（例如支持多种渲染器）
"myapp.renderers" = [
"json = mypackage.renderers:JSONRenderer",
"yaml = mypackage.renderers:YAMLRenderer",
]
3. 设备后端示例（如 vLLM 扩展）
"vllm.workers" = [
"npu = vllm_ascend.worker:AscendWorker",
]
🔔 注意：
console_scripts 是特殊 group，会自动生成可执行命令。
自定义 group 名（如 "myapp.renderers"）可任意命名，但需与使用者约定一致。

方式 2：使用 setup.py（传统方式，逐渐淘汰）

python
setup.py
from setuptools import setup, find_packages

setup(
name="my-package",
version="0.1.0",
packages=find_packages(),
entry_points={
"console_scripts": [
"mycli = mypackage.cli:main",
"mytool = mypackage.tools:run",
],
"myapp.renderers": [
"json = mypackage.renderers:JSONRenderer",
"yaml = mypackage.renderers:YAMLRenderer",
],
"vllm.workers": [
"npu = vllm_ascend.worker:AscendWorker",
],
},
)
⚠️ 注意：如果你使用 pyproject.toml，就不要再用 setup.py 定义 entry_points，避免冲突。

✅ 三、代码结构要求

假设你注册了：

toml
console_scripts = ["mycli = mypackage.cli:main"]

那么你的项目目录必须有：

my-package/
├── pyproject.toml
└── mypackage/
├── __init__.py
└── cli.py # 必须存在

且 cli.py 中要有可调用的 main 函数（或类）：

python
mypackage/cli.py
def main():
print("Hello from mycli!")
# 你的逻辑...
或者是一个 callable 类
class MyCLI:
def __call__(self):
print("Running as callable")

✅ 四、安装后效果

安装你的包后（开发模式或正式安装）：

bash
pip install -e . # 开发模式
或
pip install .

系统会自动创建命令行工具：

bash
$ mycli
Hello from mycli!

同时，其他 Python 程序可以通过 entry_points 发现你的插件：

python
from importlib.metadata import entry_points
发现所有渲染器插件
renderers = entry_points().select(group="myapp.renderers")
for ep in renderers:
RendererClass = ep.load()
renderer = RendererClass()
print(f"Loaded: {ep.name}")

✅ 五、验证 entry_points 是否注册成功
方法 1：查看已安装包的元信息

bash
pip show -f my-package grep -A 10 "entry_points"
方法 2：Python 中直接查询

python
from importlib.metadata import entry_points
查看 console_scripts
for ep in entry_points().select(group="console_scripts"):
if ep.name in ("mycli", "mytool"):
print(ep)
查看自定义 group
for ep in entry_points().select(group="myapp.renderers"):
print(ep.name, "->", ep.value)

✅ 六、常见问题
❓ Q1：entry_points 和 scripts 有什么区别？
scripts：指定一个 独立脚本文件（如 bin/myscript.sh），不经过 Python 模块系统。
console_scripts（entry_points 的一种）：指定一个 Python 可调用对象，由 setuptools 自动生成跨平台启动器（.exe / shell script）。

✅ 推荐使用 console_scripts，更便携、更 Pythonic。
❓ Q2：可以注册多个 group 吗？

可以！一个包可以同时注册 console_scripts、myapp.plugins、pytest11 等多个 group。
❓ Q3：entry_points 支持类、方法、属性吗？
✅ 支持函数、类（会被实例化？不，只是返回类对象）
✅ 支持模块中的任意可导入对象
❌ 不支持直接调用方法（如 mod:obj.method），但你可以写一个 wrapper 函数

示例：

python
mypackage/wrappers.py
from mypackage.engine import Engine

def run_engine():
engine = Engine()
engine.start()
pyproject.toml
console_scripts = ["run-engine = mypackage.wrappers:run_engine"]

✅ 总结

步骤 操作
------ ------
1️⃣ 在 pyproject.toml 的 [project.entry-points] 下定义
2️⃣ 确保目标模块和可调用对象存在
3️⃣ 安装包（pip install -e .）
4️⃣ 使用 importlib.metadata.entry_points() 动态加载

通过 entry_points，你可以轻松构建可扩展、插件化、命令行友好的 Python 应用，这是现代 Python 生态的核心实践之一。