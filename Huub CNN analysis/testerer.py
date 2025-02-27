import torch
import torch.nn as nn

class SumNet(nn.Module):
    def __init__(self):
        super(SumNet, self).__init__()

    def forward(self, a, b, c):
        d = a + b + c
        return d

class SumSumNet(nn.Module):
    def __init__(self):
        super(SumSumNet, self).__init__()

    def forward(self):
        pass


def forward_pre_hook(module, inputs):
    a, b = inputs
    return a, b * 10

def forward_hook(module, inputs, output):
    return output + 100

def main():
    sum_net = SumNet()

    sum_net.register_forward_pre_hook(forward_pre_hook)
    sum_net.register_forward_hook(forward_hook)

    a = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(2.0, requires_grad=True)
    c = torch.tensor(3.0, requires_grad=True)

    d = sum_net(a, b, c=c)

    print('d:', d)

if __name__ == '__main__':
    main()