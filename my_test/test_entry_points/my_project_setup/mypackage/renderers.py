# renderers.py 中要有可调用的 main 函数（或类）
# 或者是一个 callable 类
class JSONRenderer:
    def __call__(self):
        print("calling JSONRenderer")


class YAMLRenderer:
    def __call__(self):
        print("calling YAMLRenderer")