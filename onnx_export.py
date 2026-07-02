import argparse

import numpy as np
import torch
from stable_baselines3 import PPO

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--save", type=str, default="ppo_robot.onnx")
args = parser.parse_args()

model = PPO.load(args.model)


class OnnxablePolicy(torch.nn.Module):

    def __init__(self, policy):
        super().__init__()
        self.features_extractor = policy.features_extractor
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net

    def forward(self, observation):
        features = self.features_extractor(observation)
        latent_pi, _ = self.mlp_extractor(features)
        return self.action_net(latent_pi)


onnxable_model = OnnxablePolicy(model.policy)
onnxable_model.eval()

observation_size = 11
dummy_input = torch.randn(1, observation_size)

torch.onnx.export(
    onnxable_model,
    dummy_input,
    args.save,
    opset_version=11,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)
print("The ONNX model has been correctly exported.")
