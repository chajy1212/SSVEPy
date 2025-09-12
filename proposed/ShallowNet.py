# -*- coding:utf-8 -*-
import torch
import numpy as np
import torch.nn as nn


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


def square(x):
    return x * x


def safe_log(x, eps=1e-6):
    """ Prevents :math:`log(0)` by using :math:`log(max(x, eps))`."""
    return torch.log(torch.clamp(x, min=eps))


def np_to_var(x, requires_grad=False, dtype=None, pin_memory=False, **tensor_kwargs):
    if not hasattr(x, "__len__"):
        x = [x]
    x = np.asarray(x)
    if dtype is not None:
        x = x.astype(dtype)
    x_tensor = torch.tensor(x, requires_grad=requires_grad, **tensor_kwargs)
    if pin_memory:
        x_tensor = x_tensor.pin_memory()
    return x_tensor


class ShallowNet(nn.Module):
    def __init__(self, channel_size, input_time_length, dropout):
        super().__init__()
        cnn_kernel_1 = int(input_time_length // 22)
        cnn_kernel_2 = cnn_kernel_1 * 3

        self.cnn = nn.Sequential()
        self.cnn.add_module(
            name='block1_conv [temporal]',
            module=Conv2dWithConstraint(
                in_channels=1,
                out_channels=40,
                kernel_size=(1, cnn_kernel_1),
                stride=1
            )
        )
        self.cnn.add_module(
            name='block1_conv [spectral]',
            module=Conv2dWithConstraint(
                in_channels=40,
                out_channels=40,
                kernel_size=(channel_size, 1),
                bias=False,
                stride=1
            )
        )
        self.cnn.add_module(
            name='block1_bn',
            module=nn.BatchNorm2d(40, momentum=0.1, affine=True)
        )
        self.cnn.add_module(
            name='block1_square',
            module=Expression(square)
            # module=nn.ELU()
        )
        self.cnn.add_module(
            name='block1_average_pooling',
            module=nn.AvgPool2d(
                kernel_size=(1, cnn_kernel_2),
                stride=(1, int(cnn_kernel_2 // 5))
            )
        )
        self.cnn.add_module(
            name='block1_log',
            module=Expression(safe_log)
        )
        self.cnn.add_module(
            name='block1_dropout',
            module=nn.Dropout(dropout)
        )
        out = self.cnn(
            np_to_var(
                np.ones(
                    (1, 1, channel_size, input_time_length),
                    dtype=np.float32,
                )
            )
        )
        final_length = out.reshape(-1).shape[0]
        self.out_dim = final_length

        # self.fc = nn.Sequential()
        # self.fc.add_module(
        #     name='fully_connected',
        #     module=nn.Linear(
        #         in_features=final_length,
        #         out_features=classes
        #     )
        # )

    def forward(self, x):
        b = x.size()[0]
        # x = x.unsqueeze(dim=1)
        x = self.cnn(x)
        x = x.reshape([b, -1])
        # x = self.fc(x)
        return x


class Expression(torch.nn.Module):
    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)

    def __repr__(self):
        if hasattr(self.expression_fn, "func") and hasattr(
            self.expression_fn, "kwargs"
        ):
            expression_str = "{:s} {:s}".format(
                self.expression_fn.func.__name__, str(self.expression_fn.kwargs)
            )
        elif hasattr(self.expression_fn, "__name__"):
            expression_str = self.expression_fn.__name__
        else:
            expression_str = repr(self.expression_fn)
        return (
            self.__class__.__name__
            + "("
            + "expression="
            + str(expression_str)
            + ")"
        )