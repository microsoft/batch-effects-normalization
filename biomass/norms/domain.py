import torch
from torch import nn


# class DomainBatchNorm2d(nn.BatchNorm2d):
#     def __init__(
#         self,
#         num_features,
#         eps=1e-5,
#         momentum=0.1,
#         affine=True,
#         track_running_stats=True,
#         norm_weighting=0.75,
#     ):
#         super(DomainBatchNorm2d, self).__init__(
#             num_features, eps, momentum, affine, track_running_stats
#         )
#         self.domains = None
#         self.norm_weighting = norm_weighting

#     @classmethod
#     def from_bn2d(cls, m):
#         module = cls(m.num_features, m.eps, m.momentum, m.affine, m.track_running_stats)
#         module.load_state_dict(m.state_dict())
#         return module

#     def set_domains(self, domains):
#         self.domains = domains

#     def forward(self, input):
#         self._check_input_dim(input)

#         # if self.domains is None:
#         #     mean = input.mean([0, 2, 3])
#         #     # use biased var in train
#         #     var = input.var([0, 2, 3], unbiased=False)

#         #     input = (input - mean[None, :, None, None]) / (
#         #         torch.sqrt(var[None, :, None, None] + self.eps)
#         #     )
#         # else:
#         #     means = []
#         #     vars = []
#         #     unique_domains = self.domains.unique()
#         #     for d in unique_domains:
#         #         mask = self.domains == d
#         #         mean = input[mask].mean([0, 2, 3])
#         #         var = input[mask].var([0, 2, 3], unbiased=False)
#         #         means.append(mean)
#         #         vars.append(var)

#         #     means = torch.stack(means)
#         #     vars = torch.stack(vars)
#         #     means = (
#         #         self.norm_weighting * means + (1 - self.norm_weighting) * means[[1, 0]]
#         #     )
#         #     vars = 0.5 * vars + 0.5 * vars[[1, 0]]
#         #     for i, d in enumerate(unique_domains):
#         #         mask = self.domains == d
#         #         input[mask] = (input[mask] - means[i][None, :, None, None]) / (
#         #             torch.sqrt(vars[i][None, :, None, None] + self.eps)
#         #         )

#         #     self.mean_diff = torch.abs(means[0] - means[1]).mean()
#         #     self.var_diff = torch.abs(vars[0] - vars[1]).mean()

#         if self.domains is not None:
#             unique_domains = self.domains.unique()
#             means = []
#             vars = []
#             for d in unique_domains:
#                 mask = self.domains == d
#                 mean = input[mask].mean([0, 2, 3]).detach()
#                 var = input[mask].var([0, 2, 3], unbiased=False).detach()
#                 input[mask] = (input[mask] - mean[None, :, None, None]) / (
#                     torch.sqrt(var[None, :, None, None] + self.eps)
#                 )
#                 means.append(mean)
#                 vars.append(var)

#             self.mean_diff = torch.abs(means[0] - means[1]).mean()
#             self.var_diff = torch.abs(vars[0] - vars[1]).mean()

#         mean = input.mean([0, 2, 3])
#         # use biased var in train
#         var = input.var([0, 2, 3], unbiased=False)

#         input = (input - mean[None, :, None, None]) / (
#             torch.sqrt(var[None, :, None, None] + self.eps)
#         )

#         if self.affine:
#             input = (
#                 input * self.weight[None, :, None, None]
#                 + self.bias[None, :, None, None]
#             )

#         return input


class DomainBatchNorm2d(nn.BatchNorm2d):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(DomainBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.domains = None

    @classmethod
    def from_bn2d(cls, m):
        module = cls(m.num_features, m.eps, m.momentum, m.affine, m.track_running_stats)
        module.load_state_dict(m.state_dict())
        return module

    def set_domains(self, domains):
        self.mask = domains >= 1108

    # def weighted_mean_var(self, data):
    #     weight_sum = (
    #         torch.sum(self.weights[self.mask]) * data.shape[-1] * data.shape[-2]
    #     )
    #     mean = (
    #         torch.sum(
    #             self.weights[self.mask].view((-1, 1, 1, 1)) * data[self.mask],
    #             dim=[0, 2, 3],
    #         )
    #         / weight_sum
    #     )
    #     var = (
    #         torch.sum(
    #             self.weights[self.mask].view((-1, 1, 1, 1))
    #             * (data[self.mask] - mean[None, :, None, None]) ** 2,
    #             dim=[0, 2, 3],
    #         )
    #         / weight_sum
    #     )
    #     return mean, var

    def forward(self, input):
        self._check_input_dim(input)

        mean = input[self.mask].mean([0, 2, 3])
        #         # use biased var in train
        var = input[self.mask].var([0, 2, 3], unbiased=False)

        # mean, var = self.weighted_mean_var(input)

        input = (input - mean[None, :, None, None]) / (
            torch.sqrt(var[None, :, None, None] + self.eps)
        )
        if self.affine:
            input = (
                input * self.weight[None, :, None, None]
                + self.bias[None, :, None, None]
            )

        return input
