# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from typing import Callable, List, Union, Type

from _ark_core import _Dims, _Tensor, _NullTensor
from .data_type import DataType
from .runtime import Runtime
from .model import Model

try:
    import torch

    _no_torch = False
except ImportError:
    from . import torch_mock as torch

    _no_torch = True

NullTensor = _NullTensor


class Dims(_Dims):
    pass


Initializer = Type[Callable[[], Union[torch.Tensor, np.ndarray]]]


class Tensor:
    def __init__(
        self,
        _tensor: _Tensor,
        initializer: Initializer = None,
    ):
        """
        Initializes a new instance of the Tensor class.
        Args:
            _tensor (_ark_core._Tensor): The underlying _Tensor object.
            intializer (Initializer): The initializer for the Tensor.
        """
        self._tensor = _tensor
        self.initializer: Initializer = initializer

    def shape(self) -> List[int]:
        """
        Returns the shape of the tensor.
        """
        return self._tensor.shape().vector()

    def strides(self) -> List[int]:
        """
        Returns the strides of the tensor.
        """
        return self._tensor.strides().vector()

    def nelems(self) -> int:
        """
        Returns the number of elements in the tensor.
        """
        return self._tensor.shape().nelems()

    def dtype(self) -> DataType:
        """
        Returns the type of the tensor.
        """
        return DataType.from_ctype(self._tensor.data_type())

    def to_numpy(
        self, ndarray: np.ndarray = None, stream: int = 0
    ) -> np.ndarray:
        """
        Copy a tensor from device to host. If `ndarray` is None,
        a new numpy array will be created. If the tensor is not allocated,
        an empty numpy array without the data buffer will be returned.
        """
        np_type = self.dtype().to_numpy()
        if np_type is None:
            raise ValueError(
                f"Tensor data type {self.dtype().__name__} is not supported by numpy."
            )
        rt = Runtime.get_runtime()
        if not rt.launched():
            raise RuntimeError(
                "Tensor is not allocated yet. `Tensor.to_numpy()` is "
                "usable only after you call `Runtime.launch()`."
            )
        elif ndarray is None:
            ndarray = np.zeros(self.shape(), dtype=np_type)
        elif not ndarray.flags["C_CONTIGUOUS"]:
            raise ValueError("ndarray is not contiguous in memory")
        elif ndarray.shape != self.shape():
            raise ValueError("ndarray shape does not match the tensor")
        elif ndarray.dtype != np_type:
            raise ValueError("ndarray dtype does not match the tensor")
        elif ndarray.nbytes != self.nelems() * self.dtype().element_size():
            raise ValueError("ndarray size does not match the tensor")
        rt.executor.tensor_read(self._tensor, ndarray, stream)
        return ndarray

    def from_numpy(self, ndarray: np.ndarray, stream: int = 0) -> "Tensor":
        """
        Copies the tensor from a host numpy array to the device.
        """
        rt = Runtime.get_runtime()
        if not rt.launched():
            raise RuntimeError(
                "Tensor is not allocated yet. `Tensor.from_numpy()` is "
                "usable only after you call `Runtime.launch()`."
            )
        ndarray = ndarray.astype(self.dtype().to_numpy())
        if not ndarray.flags["C_CONTIGUOUS"]:
            ndarray = np.ascontiguousarray(ndarray)
        if ndarray.nbytes != self.nelems() * self.dtype().element_size():
            raise ValueError("ndarray size does not match the tensor")
        rt.executor.tensor_write(self._tensor, ndarray, stream)
        return self

    def to_dlpack(self):
        """
        Returns a DLPack tensor that shares the same memory with the device tensor.
        """
        rt = Runtime.get_runtime()
        if not rt.launched():
            raise RuntimeError(
                "Tensor is not allocated yet. `Tensor.to_dlpack()` is "
                "usable only after you call `Runtime.launch()`."
            )
        return rt.executor.tensor_to_dlpack(self._tensor)

    @staticmethod
    def from_dlpack(ext_tensor) -> "Tensor":
        """
        Copies the tensor from a DLPack tensor to the device.
        """
        return Tensor(_Tensor(ext_tensor))

    def to_torch(self) -> torch.Tensor:
        """
        Returns a torch tensor that shares the same memory with the device tensor.
        """
        if _no_torch:
            raise ImportError("torch is not available")
        dl_capsule = self.to_dlpack()
        torch_view = torch.utils.dlpack.from_dlpack(dl_capsule)
        # Keep dl_capsule alive not to free the memory
        torch_view.__ark_buffer__ = dl_capsule
        return torch_view

    @staticmethod
    def from_torch(tensor: torch.Tensor) -> "Tensor":
        """
        Returns an ARK tensor that shares the same memory with the torch tensor.
        """
        if _no_torch:
            raise ImportError("torch is not available")
        elif not tensor.is_contiguous():
            raise ValueError("Torch tensor must be contiguous.")
        elif tensor.device.type == "cpu":
            raise ValueError("Torch tensor must be on a device.")
        return Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(tensor))

    def copy(
        self, data: Union[np.ndarray, torch.Tensor], stream: int = 0
    ) -> "Tensor":
        """
        Copies data into this tensor. The data type may differ,
        but the size must match.
        """
        rt = Runtime.get_runtime()
        if not rt.launched():
            raise RuntimeError(
                "Tensor is not allocated yet. `Tensor.from_numpy()` is "
                "usable only after you call `Runtime.launch()`."
            )
        tensor_bytes = self.nelems() * self.dtype().element_size()
        if isinstance(data, torch.Tensor):
            if not data.is_contiguous():
                data = data.contiguous()
            if data.numel() * data.element_size() != tensor_bytes:
                raise ValueError("data size does not match the tensor")
            rt.executor.tensor_write(
                self._tensor,
                data.data_ptr(),
                tensor_bytes,
                stream,
                data.device.type == "cuda",
            )
        elif isinstance(data, np.ndarray):
            if not data.flags["C_CONTIGUOUS"]:
                data = np.ascontiguousarray(data)
            if data.nbytes != tensor_bytes:
                raise ValueError("data size does not match the tensor")
            rt.executor.tensor_write(self._tensor, data, stream)
        else:
            raise ValueError("data must be a numpy array or a torch tensor")
        return self

    def initialize(self) -> "Tensor":
        """
        Initializes the tensor.
        """
        if self.initializer is not None:
            data = self.initializer()
            self.copy(data)
        return self


class Parameter(Tensor):
    """
    A tensor as a parameter.
    """

    def __init__(self, _tensor: _Tensor):
        """
        Initializes a new instance of the Parameter class.
        """
        super().__init__(_tensor)
