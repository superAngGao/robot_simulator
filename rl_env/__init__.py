"""rl_env — RL / physics 调试环境层（Phase 2+）。

双轨设计
---------
``Env``（本文件导出）
    CPU 单环境，Simulator → CpuEngine 路径。
    用途：物理精度验证（对照 MuJoCo/Bullet）、单步交互调试、快速原型。
    不适合批量 RL 训练。

``RLEnv``（待实现，见 Q31）
    Manager-based，GpuEngine(num_envs=N) 路径。
    用途：GPU 并行 RL 训练（PPO/SAC 等）。
    前提：GpuEngine 补齐 contact forces 接口 + runtime 参数修改能力。
"""

from .base_env import Env
from .cfg import EnvCfg, NoiseCfg, ObsTermCfg
from .controllers import Controller, PDController, TorqueController
from .obs import ObsFieldSpec, ObsSchema, locomotion_obs_schema, obs_cfg_from_schema

__all__ = [
    "Env",
    "EnvCfg",
    "ObsTermCfg",
    "NoiseCfg",
    "Controller",
    "PDController",
    "TorqueController",
    "ObsFieldSpec",
    "ObsSchema",
    "locomotion_obs_schema",
    "obs_cfg_from_schema",
]
