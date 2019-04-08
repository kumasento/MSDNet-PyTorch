""" Unit tests for msdnet.py """

import unittest

import torch
import torch.nn as nn

from models.msdnet import *


class MSDNetTest(unittest.TestCase):

    def test_conv_basic(self):
        """ Test the ConvBasic module """
        mod = ConvBasic(3, 32, kernel_size=3, stride=2, padding=1)

        self.assertIsInstance(mod, nn.Module)
        self.assertIsInstance(mod.net, nn.Sequential)
        self.assertIsInstance(mod.net[0], nn.Conv2d)
        self.assertIsInstance(mod.net[1], nn.BatchNorm2d)
        self.assertIsInstance(mod.net[2], nn.ReLU)

        self.assertEqual(mod.net[0].in_channels, 3)
        self.assertEqual(mod.net[0].out_channels, 32)
        self.assertEqual(mod.net[0].stride, (2, 2))
        self.assertEqual(mod.net[0].padding, (1, 1))

    def test_msd_first_layer(self):
        """ Test the construction of the first layer in MSDNet. """
        mod = MSDFirstLayer(3, 16, grFactor=[1, 2, 4, 4])

        self.assertEqual(len(mod.layers), 4)

        # check number of output channels
        self.assertEqual(mod.layers[0].net[0].out_channels, 16)
        self.assertEqual(mod.layers[1].net[0].out_channels, 32)
        self.assertEqual(mod.layers[2].net[0].out_channels, 64)
        self.assertEqual(mod.layers[3].net[0].out_channels, 64)

        # check the correctness of computation
        x = torch.rand((1, 3, 32, 32))
        y = mod(x)
        self.assertIsInstance(y, list)
        self.assertEqual(len(y), 4)
        self.assertTrue(torch.allclose(y[0], mod.layers[0](x)))
        self.assertTrue(torch.allclose(y[1], mod.layers[1](y[0])))
        self.assertTrue(torch.allclose(y[2], mod.layers[2](y[1])))
        self.assertTrue(torch.allclose(y[3], mod.layers[3](y[2])))


if __name__ == '__main__':
    unittest.main()
