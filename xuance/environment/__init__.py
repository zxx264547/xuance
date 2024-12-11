import os
from argparse import Namespace
from xuance.environment.utils import XuanCeEnvWrapper, XuanCeMultiAgentEnvWrapper
from xuance.environment.utils import RawEnvironment, RawMultiAgentEnv
from xuance.environment.vector_envs import DummyVecEnv, DummyVecEnv_Atari, DummyVecMultiAgentEnv
from xuance.environment.vector_envs import SubprocVecEnv, SubprocVecEnv_Atari, SubprocVecMultiAgentEnv
from xuance.environment.single_agent_env import REGISTRY_ENV
from xuance.environment.multi_agent_env import REGISTRY_MULTI_AGENT_ENV
from xuance.environment.vector_envs import REGISTRY_VEC_ENV

from xuance.environment.powerSystem.IEEE13 import IEEE13


def make_envs(config: Namespace):
    def _thunk(env_seed: int = None):
        config.env_seed = env_seed
        if config.env_name in REGISTRY_ENV.keys():
            if config.env_name == "Platform":
                return REGISTRY_ENV[config.env_name](config)
            else:
                return XuanCeEnvWrapper(REGISTRY_ENV[config.env_name](config))
        elif config.env_name in REGISTRY_MULTI_AGENT_ENV.keys():
            return XuanCeMultiAgentEnvWrapper(REGISTRY_MULTI_AGENT_ENV[config.env_name](config))
        else:
            raise AttributeError(f"The environment named {config.env_name} cannot be created.")

    if config.distributed_training:
        # rank = int(os.environ['RANK'])  # for torch.nn.parallel.DistributedDataParallel
        rank = 1
        config.env_seed += rank * config.parallels

    if config.vectorize in REGISTRY_VEC_ENV.keys():
        env_fn = [_thunk for _ in range(config.parallels)]
        return REGISTRY_VEC_ENV[config.vectorize](env_fn, config.env_seed)
    elif config.vectorize == "NOREQUIRED":
        return _thunk()
    else:
        raise AttributeError(f"The vectorizer {config.vectorize} is not implemented.")
