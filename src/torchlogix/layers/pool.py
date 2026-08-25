import torch
import torch.nn.functional as F


def _to_tuple(v, n):
    if isinstance(v, tuple):
        return v
    return (v,) * n


class OrPooling2d(torch.nn.Module):
    """Logic gate based pooling layer."""

    def __init__(self, kernel_size, stride, padding=0, export_mode=False,
                 ceil_mode=False):
        super(OrPooling2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.export_mode = export_mode
        # ceil_mode keeps the ragged final window instead of discarding it, matching
        # F.max_pool*d. Needed for time axes that do not divide evenly: a 375-long
        # axis pooled by 4 is 94 windows with ceil and 93 without, and dropping that
        # window silently shortens the receptive field of every layer above it.
        self.ceil_mode = ceil_mode

    def forward(self, x):
        """Pool using logical OR (export) or max pooling (normal)."""
        if self.export_mode:
            return self._torch_or_bool(x)

        assert x.dim() == 4, "Input tensor must be 4d"

        return F.max_pool2d(
            x.float(),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
        )

    def _torch_or_bool(self, x):
        kh, kw = _to_tuple(self.kernel_size, 2)
        sh, sw = _to_tuple(self.stride, 2)
        ph, pw = _to_tuple(self.padding, 2)

        x = F.pad(x, (pw, pw, ph, ph), value=False)
        x = x.unfold(2, kh, sh).unfold(3, kw, sw).flatten(-2)  # (n, c, out_h, out_w, kh*kw)

        result = x[..., 0]
        for i in range(1, x.shape[-1]):
            result = result | x[..., i]
        return result 


    def set_export_mode(self, export_mode: bool = True):
        """Set export mode for the layer."""
        self.eval()
        self.export_mode = export_mode


class OrPooling3d(torch.nn.Module):
    """Logic gate based pooling layer."""

    def __init__(self, kernel_size, stride, padding=0, export_mode=False,
                 ceil_mode=False):
        super(OrPooling3d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.export_mode = export_mode
        # ceil_mode keeps the ragged final window instead of discarding it, matching
        # F.max_pool*d. Needed for time axes that do not divide evenly: a 375-long
        # axis pooled by 4 is 94 windows with ceil and 93 without, and dropping that
        # window silently shortens the receptive field of every layer above it.
        self.ceil_mode = ceil_mode

    def forward(self, x):

        if self.export_mode:
            return self._torch_or_pool(x)
            
        assert x.dim() == 5, "Input tensor must be 5d"

        return F.max_pool3d(
            x.float(),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
        )

    def _torch_or_pool(self, x):
        kd, kh, kw = _to_tuple(self.kernel_size, 3)
        sd, sh, sw = _to_tuple(self.stride, 3)
        pd, ph, pw = _to_tuple(self.padding, 3)

        x = F.pad(x, (pw, pw, ph, ph, pd, pd), value=False)
        x = x.unfold(2, kd, sd).unfold(3, kh, sh).unfold(4, kw, sw).flatten(-3)  # (n, c, out_d, out_h, out_w, kd*kh*kw)

        result = x[..., 0]
        for i in range(1, x.shape[-1]):
            result = result | x[..., i]
        return result

    def set_export_mode(self, export_mode: bool = True):
        """Set export mode for the layer."""
        self.eval()
        self.export_mode = export_mode
