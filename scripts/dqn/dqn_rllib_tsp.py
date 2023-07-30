from tqdm import tqdm

import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.tune.registry import register_env

from src import DeliveryEnv, CustomDQNModel

# -------------- Initialization models and environment --------------------
# Init the ray environment
ray.init(num_cpus=3, ignore_reinit_error=True, log_to_driver=False)

# Register the custom models
ModelCatalog.register_custom_model("custom_dqn_model", CustomDQNModel)

# Register the custom environment
def env_creator(env_config):
    return DeliveryEnv(n_stops=env_config['n_stops'])

register_env("DeliveryEnv-v0", env_creator)
# ------------ End Initialization models and environment -------------------


# ---------------------- Config dqn ---------------------------
config = DQNConfig()
config = config.environment(env='DeliveryEnv-v0', env_config={'n_stops': 5})
config = config.framework("torch").training(model={
    "custom_model": "custom_dqn_model",
    "fcnet_hiddens": [5]
}
)

algo = config.build()
# --------------------- End Config -----------------------------


# ---------------------- Train dqn -------------------------------
episode_reward_mean_array = []

train_steps = 18
for i in tqdm(range(train_steps)):
    result = algo.train()
    print(result['episode_reward_mean'])
    episode_reward_mean_array.append(result['episode_reward_mean'])

    if i % 5 == 0:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")
    if i+1 == train_steps:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")
# -------------------- End Train dqn ------------------------------