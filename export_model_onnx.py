"""
Export trained SB# DQN model to ONNX format, this allows us to run fast inference using ONNX runtime which reduces latency.
"""
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.policies import BasePolicy

class OnnxableSB3Policy(th.nn.Module):
    def __init__(self, policy: BasePolicy):
        super().__init__()
        self.policy = policy

    def forward(self, observation: th.Tensor) -> th.Tensor:
        return self.policy(observation)


model = DQN.load("model/best_model", device="cpu")
model.policy.eval()

onnx_policy = OnnxableSB3Policy(model.policy)

observation_size = model.observation_space.shape
dummy_input = th.randn(1, *observation_size)

th.onnx.export(
    onnx_policy,
    dummy_input,
    "model/best_model.onnx",
    opset_version=17,
    input_names=["input"],
    output_names=["output"]
)

