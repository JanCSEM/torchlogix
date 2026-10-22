"""H027: faster training/eval backends for the LUT layers (see ../backend.py).

Everything here is an exact or tolerance-bounded re-evaluation of the SAME function the
reference path (`LogicConv2d.forward`, `LogicDense.forward`, `OrPooling2d.forward`)
computes. Nothing in this package changes a model, wiring, parameter or LUT.
"""
