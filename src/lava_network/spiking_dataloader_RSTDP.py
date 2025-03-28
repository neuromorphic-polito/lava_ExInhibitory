import os
import numpy as np
import typing as ty

# Import Process level primitives
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort

# Import parent classes for ProcessModels
from lava.magma.core.model.py.model import PyLoihiProcessModel

# Import ProcessModel ports, data-types
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType

# Import execution protocol and hardware resources
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.resources import CPU

# Import decorators
from lava.magma.core.decorator import implements, requires, tag

from lava_network.spiking_dataloader import WISDM_spiking_dataloader, Py_spike_dataloader

np.set_printoptions(linewidth=np.inf)


# Do not uncomment until issue https://github.com/neuromorphic-polito/lava_ExInhibitory/issues/4 is fixed

# class WISDM_REWARD_spiking_dataloader(WISDM_spiking_dataloader):
#     """Extend the WISDM_spiking_dataloader to include a reward signal
#     The reward signal is simply the one-hot encoded vector of the current label
#     """

#     def __init__(self, signal_set, clear_intervall=0, **kwargs):
#         super().__init__(signal_set, clear_intervall, **kwargs)

#         self.spike_objcetive = OutPort(shape=(self.num_classes.init,))  # Objective spikes to the classifier

# @implements(proc=WISDM_REWARD_spiking_dataloader, protocol=LoihiProtocol)
# @requires(CPU)
# @tag("floating_pt", "fixed_pt")
# class PySpikeRewardDataloader(Py_spike_dataloader):

#     spike_objcetive: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float32, precision=32)

#     def run_spk(self):
#         super().run_spk()
#         obj_out = np.zeros(self.num_classes)

#         # Get a one hot encoded vector for the current label
#         obj_out[self.curr_label] = 1
#         self.spike_objcetive.send(obj_out)
