import torch.nn as nn


class LatentVector(nn.Module):
    def __init__(self, model, layer_name):
        super(LatentVector, self).__init__()
        self.model = model
        self.model.eval()
        self.layer_name = layer_name
        self.register_hooks()

    def register_hooks(self):
        for layer_name, layer in self.model.named_modules():
            if layer_name == self.layer_name:
                layer.register_forward_hook(self.forward_hook)

    def forward_hook(self, module, input, output):
        self.latent_vector = input

    def forward(self, x):
        prob = self.model(x)

        return self.latent_vector, prob
