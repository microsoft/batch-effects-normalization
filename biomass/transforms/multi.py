from typing import Any, Callable, Tuple, List
from torch import Tensor


class MultiView(object):
    def __init__(self, transform: Callable[[Any], Tensor], n_views: int = 1):
        self.transform = transform
        self.n_views = n_views

    def __call__(self, x: Any) -> List[Tensor]:
        return [self.transform(x) for _ in range(self.n_views)]
