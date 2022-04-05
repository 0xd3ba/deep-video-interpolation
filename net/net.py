# net.py -- Module containing the main code for the network
# Code Reference: https://github.com/HyeongminLEE/pytorch-sepconv/


import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom module imports
# import net.sepconv
import net.backbone.fe as fe
import net.backbone.rt_fe as rt_fe
import net.subnet.subnet as subnet
import net.subnet.rt_subnet as rt_subnet


# ================= CONSTANTS ===========================================
FRAME_DIM_MULTIPLE = 32     # Frame dimensions must be a multiple of this
# =======================================================================


class InterpolationNet(nn.Module):
    """ Main network that is responsible for outputting the kernels per pixel """

    def __init__(self, real_time, in_channels=64, out_channels=51):
        super().__init__()

        # Set the appropriate class references
        # TODO: Write the appropriate references for real-time variant
        BackboneNet = None if real_time else fe.FeatureExtractorNet
        SubNet = None if real_time else subnet.SubNet

        self.backbone = BackboneNet()
        self.vnet_1 = SubNet(in_channels, out_channels)
        self.hnet_1 = SubNet(in_channels, out_channels)
        self.vnet_2 = SubNet(in_channels, out_channels)
        self.hnet_2 = SubNet(in_channels, out_channels)

    def forward(self, frame_prev, frame_next):
        """
        Returns the interpolated frame between frame_prev and frame_next
        Shape of each frame: (batch, channels, height, width)
        """
        h_prev, w_prev = frame_prev.shape[2:]
        h_next, w_next = frame_next.shape[2:]

        # Some sanity checks
        assert (h_prev == h_next) and (w_prev == w_next), "Frame sizes doesn't match"

        # Pad the frames appropriately in height/width, if they are not multiples of 32
        need_h_pad = (h_prev % FRAME_DIM_MULTIPLE) != 0
        need_w_pad = (w_prev % FRAME_DIM_MULTIPLE) != 0

        # If height is not a multiple of 32, pad it so that it is
        # They will be un-padded later on, from the resulting output
        # Padding has the following semantics: (left, right, top, down) from the last dimension
        if need_h_pad:
            n_pad_pixels = FRAME_DIM_MULTIPLE - (h_prev % FRAME_DIM_MULTIPLE)
            frame_prev = F.pad(frame_prev, (0, 0, 0, n_pad_pixels))     # Pad the bottom of the frame
            frame_next = F.pad(frame_next, (0, 0, 0, n_pad_pixels))     # Pad the bottom of the frame

        # If the width is not a multiple of 32, pad it so that is is
        # They will be un-padded from the resulting output
        if need_w_pad:
            n_pad_pixels = FRAME_DIM_MULTIPLE - (w_prev % FRAME_DIM_MULTIPLE)
            frame_prev = F.pad(frame_prev, (0, n_pad_pixels, 0, 0))     # Pad the right part of the frame
            frame_next = F.pad(frame_next, (0, n_pad_pixels, 0, 0))     # Pad the right part of the frame

        # Now extract the features from the frames
        # Then send them to the corresponding subnets
        output_features = self.backbone(frame_prev, frame_next)
        k1_v = self.vnet_1(output_features)
        k1_h = self.hnet_1(output_features)
        k2_v = self.vnet_2(output_features)
        k2_h = self.hnet_2(output_features)

        # TODO: Pass the branched result to separable-convolution layer
        # TODO: Add the result to get the final interpolated frame
        # TODO: Remove the padded pixels
        # TODO: Return the frame



