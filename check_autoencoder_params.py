import inspect
from pyod.models.auto_encoder import AutoEncoder

# 打印AutoEncoder类的参数信息
print("AutoEncoder类的参数信息:")
print(inspect.signature(AutoEncoder.__init__))

# 打印AutoEncoder类的文档字符串
print("\nAutoEncoder类的文档字符串:")
print(AutoEncoder.__doc__)