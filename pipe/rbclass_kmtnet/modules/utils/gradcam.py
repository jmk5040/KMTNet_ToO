import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM(nn.Module):
    def __init__(self, model, layer_name):
        super(GradCAM, self).__init__()
        self.model = model
        self.model.eval()
        self.layer_name = layer_name
        self.register_hooks()

    def register_hooks(self):
        for layer_name, layer in self.model.named_children():
            if layer_name == self.layer_name:
                layer.register_forward_hook(self.forward_hook)
                layer.register_full_backward_hook(self.backward_hook)

    def forward(self, input, return_prob=False):
        self.model.zero_grad()

        h, w = input.shape[-2:]
        prob = self.model(input)

        prob.backward(retain_graph=True)
        weights = torch.mean(self.backward_output, dim=(1, 2), keepdim=True)
        out = torch.sum(weights * self.forward_output, dim=0)
        out = torch.relu(out) / torch.max(out)
        out = F.interpolate(
                out.unsqueeze(0).unsqueeze(0),
                (h, w),
                mode='bilinear',
                align_corners=True
            )

        if return_prob:
            return out.cpu().detach().squeeze().numpy(), prob.cpu().detach().squeeze().numpy()

        else:
            return out.cpu().detach().squeeze().numpy()

    def forward_hook(self, module, input, output):
        self.forward_output = torch.squeeze(output)

    def backward_hook(self, module, grad_input, grad_output):
        self.backward_output = torch.squeeze(grad_output[0])
