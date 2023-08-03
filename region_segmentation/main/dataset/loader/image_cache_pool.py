from utils import Rotator
from .cache import query, register


def gen(func: callable, arg: str) -> callable:
    def proxy():
        # print('loading ... ')
        return func(arg)
    return proxy


class ImageLoader(object):
    def __init__(self, path: str, zoom: float, in_memory: bool = False, **kwargs):
        self.path = path
        self.zoom = zoom
        self.in_memory = in_memory
        if in_memory:
            register(path, gen(Rotator.ImageCropper, path))

    def __call__(self, grid: dict):
        if self.in_memory:
            loader = query(self.path)
        else:
            loader = Rotator.ImageCropper(self.path)
        # loader = Rotator.ImageCropper(self.path)
        return loader.get(
            site=(grid['x'] * self.zoom, grid['y'] * self.zoom),
            size=(grid['w'], grid['h']),
            degree=grid['degree'],
            scale=grid['scaling'] * self.zoom,
        )
