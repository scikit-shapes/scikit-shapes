from .neighborhoods import Neighborhoods


class TrivialNeighborhoods(Neighborhoods):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_trivial = True

    def convolve(self, *, signal):
        return signal
