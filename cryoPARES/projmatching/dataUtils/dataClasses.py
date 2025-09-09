import torch


class ImageTensor(torch.Tensor):

    @property
    def pixel_size(self):
        return getattr(self, "_pixel_size", None)

    @pixel_size.setter
    def pixel_size(self, angpix:float):
        setattr(self, "_pixel_size", angpix)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):

        if kwargs is None:
            kwargs = {}

        # Handle string conversion functions separately
        if func in (torch.Tensor.__repr__, torch.Tensor.__str__):
            return super(torch.Tensor, args[0]).__repr__()  # Directly call the superclass method
        result = super().__torch_function__(func, types, args, kwargs)

        # Convert the result to an ImageTensor, if it's a tensor
        if isinstance(result, torch.Tensor):
            result = ImageTensor(result)
            if isinstance(args[0], ImageTensor):
                result.pixel_size = args[0].pixel_size

        return result

    def __repr__(self, *, tensor_contents=None):
        temp_tensor = torch.Tensor(self)
        return "ImageT"+repr(temp_tensor)[1:]
