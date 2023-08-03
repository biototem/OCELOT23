# TODO: 事实表明，流程这东西会越弄越多，所以早晚有一天，要给它改成动态 import 的写法，以免各流程间的预备状态相互干扰或拖累
from .build import do_build
from .train import do_train
# from .valid import do_valid
from .evaluate import do_calculate, do_evaluate, Visualizer




__all__ = [
    'do_build',
    'do_train',
    'do_calculate',
    'do_evaluate',
]
