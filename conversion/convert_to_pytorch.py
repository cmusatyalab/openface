#!/usr/bin/env python3

import numpy as np
import torch
import torchfile

import openface


def get_param_tensor(layer_dict, param_name):
    return torch.from_numpy(layer_dict[param_name]).float()


def copy_conv_layer_params(from_layer, to_layer, reshape_weight=None):
    weight = get_param_tensor(from_layer, b'weight')
    if reshape_weight is not None:
        weight = torch.reshape(weight, reshape_weight)
    bias = get_param_tensor(from_layer, b'bias')
    to_layer.weight.copy_(weight)
    to_layer.bias.copy_(bias)


def copy_bn_layer_params(from_layer, to_layer):
    to_layer.weight.copy_(get_param_tensor(from_layer, b'weight'))
    to_layer.bias.copy_(get_param_tensor(from_layer, b'bias'))
    to_layer.running_mean.copy_(get_param_tensor(from_layer, b'running_mean'))
    to_layer.running_var.copy_(get_param_tensor(from_layer, b'running_var'))


def copy_inception_params(from_modules, to_modules, conv_layers_indices, bn_layers_indices):
    for branch_ind, layer_ind in conv_layers_indices:
        copy_conv_layer_params(from_modules[branch_ind][layer_ind], to_modules[branch_ind][layer_ind])
    for branch_ind, layer_ind in bn_layers_indices:
        copy_bn_layer_params(from_modules[branch_ind][layer_ind], to_modules[branch_ind][layer_ind])


if __name__ == '__main__':
    openface_model = openface.OpenFaceNet()

    # Load weights from lua torch model
    # Load the pretrained model first with the latest LuaTorch and re-save it
    lua_model = torchfile.load('nn4.small2.v1.resaved.t7')
    lua_model_layers = lua_model[b'modules']

    with torch.no_grad():
        copy_conv_layer_params(lua_model_layers[0], openface_model.conv1, (64, 3, 7, 7))
        copy_bn_layer_params(lua_model_layers[1], openface_model.bn1)
        copy_conv_layer_params(lua_model_layers[5], openface_model.conv2, (64, 64, 1, 1))
        copy_bn_layer_params(lua_model_layers[6], openface_model.bn2)
        copy_conv_layer_params(lua_model_layers[8], openface_model.conv3, (192, 64, 3, 3))
        copy_bn_layer_params(lua_model_layers[9], openface_model.bn3)

        incept3a_modules = [branch[b'modules'] for branch in lua_model_layers[13][b'modules'][0][b'modules']]
        copy_inception_params(incept3a_modules, openface_model.incept3a.branches,
                              conv_layers_indices=((0, 0), (0, 3), (1, 0), (1, 3), (2, 1), (3, 0)),
                              bn_layers_indices=((0, 1), (0, 4), (1, 1), (1, 4), (2, 2), (3, 1)))

        incept3b_modules = [branch[b'modules'] for branch in lua_model_layers[14][b'modules'][0][b'modules']]
        copy_inception_params(incept3b_modules, openface_model.incept3b.branches,
                              conv_layers_indices=((0, 0), (0, 3), (1, 0), (1, 3), (2, 1), (3, 0)),
                              bn_layers_indices=((0, 1), (0, 4), (1, 1), (1, 4), (2, 2), (3, 1)))

        incept3c_modules = [branch[b'modules'] for branch in lua_model_layers[15][b'modules'][0][b'modules']]
        copy_inception_params(incept3c_modules, openface_model.incept3c.branches,
                              conv_layers_indices=((0, 0), (0, 3), (1, 0), (1, 3)),
                              bn_layers_indices=((0, 1), (0, 4), (1, 1), (1, 4)))

        incept4a_modules = [branch[b'modules'] for branch in lua_model_layers[16][b'modules'][0][b'modules']]
        copy_inception_params(incept4a_modules, openface_model.incept4a.branches,
                              conv_layers_indices=((0, 0), (0, 3), (1, 0), (1, 3), (2, 1), (3, 0)),
                              bn_layers_indices=((0, 1), (0, 4), (1, 1), (1, 4), (2, 2), (3, 1)))

        incept4e_modules = [branch[b'modules'] for branch in lua_model_layers[17][b'modules'][0][b'modules']]
        copy_inception_params(incept4e_modules, openface_model.incept4e.branches,
                              conv_layers_indices=((0, 0), (0, 3), (1, 0), (1, 3)),
                              bn_layers_indices=((0, 1), (0, 4), (1, 1), (1, 4)))

        incept5a_modules = [branch[b'modules'] for branch in lua_model_layers[18][b'modules'][0][b'modules']]
        copy_inception_params(incept5a_modules, openface_model.incept5a.branches,
                              conv_layers_indices=((0, 0), (0, 3),(1, 1), (2, 0)),
                              bn_layers_indices=((0, 1), (0, 4), (1, 2), (2, 1)))

        incept5b_modules = [branch[b'modules'] for branch in lua_model_layers[20][b'modules'][0][b'modules']]
        copy_inception_params(incept5b_modules, openface_model.incept5b.branches,
                              conv_layers_indices=((0, 0), (0, 3),(1, 1), (2, 0)),
                              bn_layers_indices=((0, 1), (0, 4), (1, 2), (2, 1)))

        openface_model.ln.weight.copy_(get_param_tensor(lua_model_layers[24], b'weight'))
        openface_model.ln.bias.copy_(get_param_tensor(lua_model_layers[24], b'bias'))

    # Run forward pass
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    openface_model = openface_model.to(device)
    openface_model.eval()
    ones = torch.ones((1, 3, 96, 96), dtype=torch.float32)
    ones = ones.to(device)
    pytorch_out = openface_model(ones).squeeze(0)

    # Compare results with lua torch model results
    lua_out = torchfile.load('l26out.t7')

    np.testing.assert_allclose(pytorch_out.cpu().detach().numpy(), lua_out, rtol=1e-03, atol=1e-05)

    # Save model state dict
    openface_model.to(torch.device('cpu'))
    torch.save(openface_model.state_dict(), 'nn4.small2.v1.pt')
