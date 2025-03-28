from lava.proc.dense.process import Dense
from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.dense.models import AbstractPyDenseModelFloat, PyDenseModelFloat
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires , tag
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType

import numpy as np
import typing as ty


class DenseEncoder(Dense):
    pass

@implements(proc=DenseEncoder, protocol=LoihiProtocol)
@requires(CPU)
@tag("fixed_pt", "floating_pt")
class PyDenseEncoderModelFloat(AbstractPyDenseModelFloat):
    """Same as AbstractPyDenseModelFloat. This class is necessary because of
    Lava limitation to choose the different models for the same process.
    This dense is just for the encoding. It works on Float and outputs float,
    because the conversion to fixed-point is done in the following LIF"
    """
    pass
