import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import override
from ray.rllib.utils.torch_utils import FLOAT_MIN


class CustomDQNModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)

        input_size = obs_space.original_space["real_obs"].shape[0]
        self.input_layer = nn.Linear(input_size, 256)
        self.hidden1 = nn.Linear(256, 256)
        self.hidden2 = nn.Linear(256, 256)
        self.output_layer = nn.Linear(256, num_outputs)

        # Add a separate output head for the value function
        self.value_output_layer = nn.Linear(num_outputs, 1)  # Update this line

    def forward(self, input_dict, state, seq_lens):
        action_mask = input_dict["obs"]["action_mask"]

        x = input_dict["obs"]["real_obs"].float()
        x = torch.relu(self.input_layer(x))
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output_layer(x)

        # Compute the value function
        self.value = self.value_output_layer(x)  # Update this line

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = x + inf_mask

        return masked_logits, []

    @override(TorchModelV2)
    def value_function(self):
        return torch.reshape(self._value, [-1])