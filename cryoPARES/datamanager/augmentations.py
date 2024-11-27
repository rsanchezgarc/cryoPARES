from abc import ABC, abstractmethod


class AugmenterBase(ABC):
    @abstractmethod
    def applyAugmentation(self, imgs, degEulerList, shiftFractionList):
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()