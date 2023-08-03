from utils import Rotator


class ImageLoader(object):
    def __init__(self, path: str, zoom: float, in_memory: bool = False, **kwargs):
        self.path = path
        self.zoom = zoom
        self.in_memory = in_memory
        self.cache = None

    def load(self) -> Rotator.ImageCropper:
        if self.in_memory:
            if self.cache is None:
                self.cache = Rotator.ImageCropper(self.path)
            return self.cache
        else:
            return Rotator.ImageCropper(self.path)

    def __call__(self, grid: dict):
        loader = self.load()
        return loader.get(
            site=(grid['x'] * self.zoom, grid['y'] * self.zoom),
            size=(grid['w'], grid['h']),
            degree=grid['degree'],
            scale=grid['scaling'] * self.zoom,
        )
