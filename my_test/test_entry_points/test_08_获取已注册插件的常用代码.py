from importlib.metadata import entry_points

# todo 1、查看所有已安装包的 entry_points
# 获取所有 entry_points（返回 EntryPoints 对象，可迭代）
eps = entry_points()

# 按 group 查看（最常见的是 'console_scripts'）
console_scripts = eps.select(group='console_scripts')
for ep in console_scripts:
    print(ep.name, "=>", ep.value)

'''
pip => pip._internal.cli.main:main
pip3 => pip._internal.cli.main:main
isympy => isympy:main
cygdb => Cython.Debugger.Cygdb:main
cython => Cython.Compiler.Main:setuptools_main
cythonize => Cython.Build.Cythonize:main
datamodel-codegen => datamodel_code_generator.__main__:main
pyproject-build => build.__main__:entrypoint
yapf => yapf:run_main
yapf-diff => yapf_third_party.yapf_diff.yapf_diff:main
pyrsa-decrypt => rsa.cli:decrypt
pyrsa-encrypt => rsa.cli:encrypt
pyrsa-keygen => rsa.cli:keygen
pyrsa-priv2pub => rsa.util:private_to_public
pyrsa-sign => rsa.cli:sign
pyrsa-verify => rsa.cli:verify
nltk => nltk.cli:cli
tabulate => tabulate:_main
torchfrtrace => tools.flight_recorder.fr_trace:main
torchrun => torch.distributed.run:main
fastavro => fastavro.__main__:main
rouge => bin.rouge_cmd:main
py.test => pytest:console_main
pytest => pytest:console_main
activate-global-python-argcomplete => argcomplete.scripts.activate_global_python_argcomplete:main
python-argcomplete-check-easy-install-script => argcomplete.scripts.python_argcomplete_check_easy_install_script:main
register-python-argcomplete => argcomplete.scripts.register_python_argcomplete:main
black => black:patched_main
blackd => blackd:patched_main [d]
evaluate-cli => evaluate.commands.evaluate_cli:main
isort => isort.main:main
isort-identify-imports => isort.main:identify_imports_main
torch_npu_run => torch_npu.distributed.run:_main
ais_bench => ais_bench.benchmark.cli.main:main
plotly_get_chrome => plotly.io._kaleido:plotly_get_chrome
pyjson5 => json5.tool:main
pygrun => antlr4._pygrun:main
vllm => vllm.entrypoints.cli.main:main
python-grpc-tools-protoc => grpc_tools.protoc:entrypoint
normalizer => charset_normalizer.cli:cli_detect
datasets-cli => datasets.commands.datasets_cli:main
genson => genson.__main__:main
evaluate_functional_correctness => human_eval.evaluate_functional_correctness:main
bfcl => bfcl_eval.__main__:cli
sacrebleu => sacrebleu.sacrebleu:main
fonttools => fontTools.__main__:main
pyftmerge => fontTools.merge:main
pyftsubset => fontTools.subset:main
ttx => fontTools.ttx:main
hf => huggingface_hub.cli.hf:main
huggingface-cli => huggingface_hub.commands.huggingface_cli:main
tiny-agents => huggingface_hub.inference._mcp.cli:app
accelerate => accelerate.commands.accelerate_cli:main
accelerate-config => accelerate.commands.config:main
accelerate-estimate-memory => accelerate.commands.estimate:main
accelerate-launch => accelerate.commands.launch:main
accelerate-merge-weights => accelerate.commands.merge:main
rouge_chinese => bin.rouge_cmd:main
tb-gcp-uploader => google.cloud.aiplatform.tensorboard.uploader_main:run_main
modelscope => modelscope.cli.cli:run_cmd
ray => ray.scripts.scripts:main
serve => ray.serve.scripts:cli
tune => ray.tune.cli.scripts:cli
flask => flask.cli:main
f2py => numpy.f2py.f2py2e:main
hypercorn => hypercorn.__main__:main
wheel => wheel.cli:main
setuptools-scm => setuptools_scm._cli:main
quart => quart.cli:main
ccmake => cmake:ccmake
cmake => cmake:cmake
cpack => cmake:cpack
ctest => cmake:ctest
pybind11-config => pybind11.__main__:main
pybase64 => pybase64.__main__:main
dotenv => dotenv.__main__:cli
openai => openai.cli:main
jsonschema => jsonschema.cli:main
tqdm => tqdm.cli:main
websockets => websockets.cli:main
gguf-convert-endian => gguf.scripts.gguf_convert_endian:main
gguf-dump => gguf.scripts.gguf_dump:main
gguf-editor-gui => gguf.scripts.gguf_editor_gui:main
gguf-new-metadata => gguf.scripts.gguf_new_metadata:main
gguf-set-metadata => gguf.scripts.gguf_set_metadata:main
distro => distro.distro:main
json-playground => partial_json_parser.playground:main
cbor2 => cbor2.tool:main
httpx => httpx:main
markdown-it => markdown_it.cli.parse:main
uvicorn => uvicorn.main:main
email_validator => email_validator.__main__:main
watchfiles => watchfiles.cli:cli
typer => typer.cli:main
fastapi => fastapi.cli:main
transformers => transformers.commands.transformers_cli:main
transformers-cli => transformers.commands.transformers_cli:main_cli
pygmentize => pygments.cmdline:main
mistral_common => mistral_common.experimental.app.main:cli
cpuinfo => cpuinfo:main

'''

print(f'\n\n\n')

# 查找vllm-ascend注册的entry_points
console_scripts = eps.select(group='vllm.platform_plugins')
for ep in console_scripts:
    print(ep.name, "=====>", ep.value)

'''
ascend =====> vllm_ascend:register
'''


# todo 2、加载并调用一个 entry_point
# from importlib.metadata import entry_points
#
# # 获取名为 'vllm' 的 console_script
# ep = entry_points().select(group='console_scripts', name='vllm').__next__()
#
# # 加载对应的可调用对象（函数/类）
# main_func = ep.load()  # 返回 vllm.scripts.main 函数
#
# # 调用它（通常用于测试或嵌入）
# # main_func()  # 相当于运行 `vllm` 命令



# todo 3、自定义 group（用于插件系统）
# 假设你在开发一个支持插件的框架：
#
# 步骤 1：插件包注册 entry_point
# 在插件包的 pyproject.toml 中：
[project.entry-points."myapp.renderers"]
json = "myplugin.renderers:JSONRenderer"
html = "myplugin.renderers:HTMLRenderer"




