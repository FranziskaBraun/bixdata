import math
import torch
from torch import nn


class GatedMultimodalUnit(nn.Module):
    def __init__(self, input_dim_1, input_dim_2, hidden_dim):
        """
        Initializes the Gated Multimodal Unit (GMU).

        :param input_dim_1: Dimensionality of the first modality
        :param input_dim_2: Dimensionality of the second modality
        :param hidden_dim: Dimensionality of the transformed output
        """
        super(GatedMultimodalUnit, self).__init__()

        # Linear transformations for each modality
        self.fc_x1 = nn.Linear(input_dim_1, hidden_dim)
        self.fc_x2 = nn.Linear(input_dim_2, hidden_dim)

        # Gating mechanism
        self.gate_x1 = nn.Linear(input_dim_1, hidden_dim)
        self.gate_x2 = nn.Linear(input_dim_2, hidden_dim)

    def forward(self, x1, x2):
        """
        Forward pass for GMU.

        :param x1: Tensor of shape (batch_size, input_dim_1) representing modality 1
        :param x2: Tensor of shape (batch_size, input_dim_2) representing modality 2
        :return: Fused representation of shape (batch_size, hidden_dim)
        """
        # Compute transformed representations
        h1 = torch.tanh(self.fc_x1(x1))
        h2 = torch.tanh(self.fc_x2(x2))

        # Compute gate values
        g = torch.sigmoid(self.gate_x1(x1) + self.gate_x2(x2))

        # Compute final output
        h = g * h1 + (1 - g) * h2  # Element-wise weighted sum

        return h


class GatedMultimodalLayer(nn.Module):
    """ Gated Multimodal Layer based on 'Gated multimodal networks, Arevalo1 et al.' (https://arxiv.org/abs/1702.01992) """
    def __init__(self, size_in1, size_in2, size_out):
        super().__init__()
        self.size_in1, self.size_in2, self.size_out = size_in1, size_in2, size_out

        # Weights hidden state modality 1
        weights_hidden1 = torch.Tensor(size_out, size_in1)
        self.weights_hidden1 = nn.Parameter(weights_hidden1)

        # Weights hidden state modality 2
        weights_hidden2 = torch.Tensor(size_out, size_in2)
        self.weights_hidden2 = nn.Parameter(weights_hidden2)

        # Weight for sigmoid
        weight_sigmoid = torch.Tensor(size_out*2)
        self.weight_sigmoid = nn.Parameter(weight_sigmoid)

        # initialize weights
        nn.init.kaiming_uniform_(self.weights_hidden1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weights_hidden2, a=math.sqrt(5))

        # Activation functions
        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x1, x2):
        h1 = self.tanh_f(torch.mm(x1, self.weights_hidden1.t()))
        h2 = self.tanh_f(torch.mm(x2, self.weights_hidden2.t()))
        x = torch.cat((h1, h2), dim=1)
        z = self.sigmoid_f(torch.matmul(x, self.weight_sigmoid.t()))

        return z.view(z.size()[0],1)*h1 + (1-z).view(z.size()[0],1)*h2


class ConcatModel(nn.Module):

    def __init__(self, ft1_size=768, ft2_size=768, output_dim=2, mlp_dropout=0.1):

        super(ConcatModel, self).__init__()
        self.drop1 = nn.Dropout(p=mlp_dropout)
        self.linear1 = nn.Linear(ft1_size+ft2_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.drop2 = nn.Dropout(p=mlp_dropout)
        self.linear2 = nn.Linear(512, output_dim)

    def forward(self, x1, x2):

        x = torch.cat((x1, x2), dim=1)
        x = self.drop1(x)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop2(x)

        return self.linear2(x)


class GatedModel(nn.Module):

    def __init__(self, ft1_size=768, ft2_size=768, output_dim=2, mlp_dropout=0.1):

        super(GatedModel, self).__init__()
        self.gated_linear1 = GatedMultimodalLayer(ft1_size, ft2_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=mlp_dropout)
        self.linear1 = nn.Linear(512, output_dim)

    def forward(self, x1, x2):

        x = self.gated_linear1(x1, x2)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        return self.linear1(x)


class SingleModel(nn.Module):

    def __init__(self, ft1_size=768, hidden_dim=512, output_dim=2, mlp_dropout=0.1):

        super(SingleModel, self).__init__()
        self.drop1 = nn.Dropout(p=mlp_dropout)
        self.linear1 = nn.Linear(ft1_size, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.drop2 = nn.Dropout(p=mlp_dropout)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.drop1(x)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop2(x)
        return self.linear2(x)
