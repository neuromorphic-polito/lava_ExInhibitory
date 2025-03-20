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

np.set_printoptions(linewidth=np.inf)


class WisdmDatasetParser():
    """Split the given file into training, validation and test datasets.
    For each one decide if do shuffle or not.

    The percentages for the split are already defined in the dataset.

    Same as the snntorch_ExInhibitory version.
    """
    def __init__(self,
                 file_name: str,
                 norm: str = "std",
                 class_sublset: ty.Optional[str] = None,
                 subset_list: ty.Optional[ty.List[int]] = None):

        """Initialize the class.

        Parameters
        ----------

        file_name : str
            The path to the file containing the dataset.

        norm : str
            The normalization to apply to the dataset. It can be "std" for
            standard normalization, custom to just divide by the standard
            deviation or None to not normalize the dataset.

        class_sublset : str
            The subset of classes to use. It can be "7BC" for the 7 best
            classes or "7WC" for the 7 worst classes in terms of separability
            score. It can also be "subset_2" for a custom subset of classes
            identified by Vittorio, or custom to give use the custom
            subset list.

        subset_list : list[int]
            The list of classes to use if class_subset is set to custom.
        """
        self.file_name = file_name
        (x_train, x_val, x_test, y_train, y_val, y_test) = self.load_wisdm2_data(file_name)
        self.class_sublset = class_sublset
        self.norm = norm
        self.mean = np.mean(x_train, axis=(0,1))
        self.std = np.std(x_train, axis=(0,1))
        print(self.mean.shape)
        print(self.std.shape)
        if self.norm == "std":
            x_train = x_train - self.mean
            x_train = x_train/self.std

            x_val = x_val - self.mean
            x_val = x_val/self.std

            x_test = x_test - self.mean
            x_test = x_test/self.std

        elif self.norm == "custom":
            x_train = x_train/self.std
            x_val = x_val/self.std
            x_test = x_test/self.std

        elif self.norm == None:
            pass

        print(f'ytrain shape {y_train.shape}')
        print(f'yval shape {y_val.shape}')
        print(f'ytest shape {y_test.shape}')

        x_train = np.transpose(x_train,axes=(0,2,1))
        x_val = np.transpose(x_val,axes=(0,2,1))
        x_test = np.transpose(x_test,axes=(0,2,1))

        if self.class_sublset is not None:
            if self.class_sublset == '7BC':
                selected_classes =  [1,6,7,8,13,14,17]
            elif self.class_sublset == '7WC':
                selected_classes = [11,12,13,14,15,16,17]
            elif self.class_sublset == 'subset_2':
                selected_classes = [6, 7, 8, 9, 10, 11, 12]
            elif self.class_sublset == 'custom':
                selected_classes = subset_list

            x_train, y_train = filter_dataset(x_train, y_train, selected_classes)
            x_val, y_val = filter_dataset(x_val, y_val, selected_classes)
            x_test, y_test = filter_dataset(x_test, y_test, selected_classes)

        if len(y_test.shape) > 1:
            self.train_dataset = (x_train, np.argmax(y_train, axis=-1))
            self.val_dataset = (x_val, np.argmax(y_val, axis=-1))
            self.test_dataset = (x_test, np.argmax(y_test, axis=-1))
        else:
            self.train_dataset = (x_train, y_train)
            self.val_dataset = (x_val,y_val)
            self.test_dataset = (x_test,y_test)

        print(f'num classes train dataset: {self.train_dataset[1].max()+1} occurrences of each class:{np.bincount(self.train_dataset[1])}')
        print(f'num classes eval dataset: {self.val_dataset[1].max()+1} occurrences of each class:{np.bincount(self.val_dataset[1])}')
        print(f'num classes test dataset: {self.test_dataset[1].max()+1} occurrences of each class:{np.bincount(self.test_dataset[1])}')

    def get_training_set(self, subset=None, shuffle=True):

        if subset:
            N = self.test_dataset[0].shape[0]

            if shuffle:
                ids = np.array(range(0, N))
                np.random.shuffle(ids)
                ids = ids[:subset]

            else:
                ids = np.array(range(0, subset))

            return np.array(self.train_dataset[0][ids]), np.array(self.train_dataset[1][ids])
        return self.train_dataset

    def get_validation_set(self, subset=None, shuffle=True):

        if subset:
            N = self.test_dataset[0].shape[0]

            if shuffle:
                ids = np.array(range(0, N))
                np.random.shuffle(ids)
                ids = ids[:subset]

            else:
                ids = np.array(range(0, subset))

            return np.array(self.val_dataset[0][ids]), np.array(self.val_dataset[1][ids])

        return self.val_dataset

    def get_test_set(self, subset=None, shuffle=True):

        if subset:
            N = self.test_dataset[0].shape[0]

            if shuffle:
                ids = np.array(range(0, N))
                np.random.shuffle(ids)
                ids = ids[:subset]

            else:
                ids = np.array(range(0, subset))

            return np.array(self.test_dataset[0][ids]), np.array(self.test_dataset[1][ids])

        return self.test_dataset

    def de_std(self, data):
        if self.norm == "norm":
            data= data * self.std
            data= data + self.mean
        if self.norm == "custom":
            data= data * self.std

    def do_std(self, data):
        data= data - self.mean
        data= data / self.std

    @staticmethod
    def load_wisdm2_data(file_path):
        filepath = os.path.join(file_path)
        data = np.load(filepath)
        return (data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3'], data['arr_4'], data['arr_5'])


def filter_dataset(x_train, y_train, selected_classes):
    # Create a mapping dictionary for the selected classes
    class_mapping = {original: new for new, original in enumerate(selected_classes)}

    # Convert selected_classes to a set for faster look-up
    selected_set = set(selected_classes)

    # Get the indices of the selected classes in y_train
    original_class_indices = np.argmax(y_train, axis=1)
    mask = np.isin(original_class_indices, selected_classes)

    # Filter the data and labels using the mask
    filtered_x = x_train[mask]
    filtered_y = y_train[mask]

    # Map the original class indices to new indices
    new_class_indices = np.vectorize(class_mapping.get)(original_class_indices[mask])

    # Create the new one-hot encoded labels
    new_one_hot_y = np.zeros((filtered_y.shape[0], len(selected_classes)))
    new_one_hot_y[np.arange(filtered_y.shape[0]), new_class_indices] = 1

    return filtered_x, new_one_hot_y


class WISDM_spiking_dataloader(AbstractProcess):
    """Spiking dataloader for the WISDM dataset.

    This process is responsible for sending the input samples to the network
    and the ground truth to the output process.

    Works only on CPU.

    This does not work on spikes, it works on floating point values.
    """

    def __init__(self,
                 signal_set: list[ty.Any],
                 clear_intervall: int = 0,
                 **kwargs):
        """Init of the class.

        Parameters
        ----------
        signal_set : list[ty.Any]
            The result from the WisdmDatasetParser class.
        clear_intervall : int, optional
            How may timesteps are required for the input to propagate all the
            way through the network. Should be set to the number of layers of
            the network between input and output, by default 0
        """

        super().__init__()

        data_shape = signal_set[0].shape
        num_samples = data_shape[0]
        num_channels = data_shape[1]
        num_timesteps = data_shape[2]
        num_classes = signal_set[1].max()+1

        self.clear_intervall = Var(shape=(1,), init=clear_intervall)  # Network delay
        self.samples = Var(shape=(num_samples, num_channels, num_timesteps), init=signal_set[0])  # Input samples
        self.labels = Var(shape=(num_samples,), init=signal_set[1])  # Ground truth labels

        self.data_out = OutPort(shape=(num_channels,))  # Input spikes to the classifier
        self.label_out = OutPort(shape=(1,))  # Ground truth labels to OutputProc

        self.curr_sample = Var(shape=(num_channels, num_timesteps))  # Current sample being processed
        self.curr_label = Var(shape=(1,), init=signal_set[1][0])
        self.num_samples = Var(shape=(1,), init=num_samples)
        self.num_timesteps_per_sample = Var(shape=(1,), init=num_timesteps)
        self.num_classes= Var(shape=(1,), init=num_classes)
        self.curr_sample_time_step = Var(shape=(1,))


@implements(proc=WISDM_spiking_dataloader, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt", "fixed_pt")
class Py_spike_dataloader(PyLoihiProcessModel):

    clear_intervall: int = LavaPyType(int, int, precision=32)
    num_samples: int = LavaPyType(int, int, precision=32)
    samples: np.ndarray = LavaPyType(np.ndarray, np.float32, precision=32)
    labels: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=32)

    data_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float32, precision=32)
    label_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float32, precision=32)
    num_timesteps_per_sample: int = LavaPyType(int, int, precision=32)
    num_classes: int = LavaPyType(int, int, precision=32)
    curr_label: int = LavaPyType(int, int, precision=32)
    curr_sample: np.ndarray = LavaPyType(np.ndarray, np.float32, precision=32)
    curr_sample_time_step: int = LavaPyType(int, int, precision=32)

    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)
        self.curr_sample_id = 0
        self.curr_sample_time_step = 0

    def post_guard(self):
        """Check if it is time to go to the next sample."""
        if self.time_step % (self.num_timesteps_per_sample + self.clear_intervall) == 0:
            return True
        return False

    def run_post_mgmt(self):
        """Do what is required before going to the next sample,
        Reset time step counter, send the label, go to next id"""
        self.curr_sample_time_step = 0
        self.curr_label = self.labels[self.curr_sample_id]
        self.label_out.send(np.array([self.curr_label]))
        self.curr_sample_id += 1

    def run_spk(self):
        """At each time step, send a sample of the signal to the network.
        """
        # Check if there is data to send
        if self.curr_sample_time_step < self.num_timesteps_per_sample:
            s_out = self.samples[
                self.curr_sample_id,:, self.curr_sample_time_step]
            self.curr_sample_time_step += 1

        # Since the sample is finished, we present zeros until it is time to
        # reset and go to the next sample.
        else:
            s_out = np.array([0.0]*self.curr_sample.shape[0])

        # Send data
        self.data_out.send(np.array([s_out]).flatten().astype(float))
