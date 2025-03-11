import torch
import torch.nn as nn
from monai.networks.nets import UNet


class TruncatedUNet(nn.Module):
    """
    A modified UNet that returns the output of a specific intermediate layer.
    """

    def __init__(self, unet_params, layer_depth=-2):
        """
        Args:
            original_unet: The original UNet model
            layer_depth: Which layer to return (-2 for second-to-last, -3 for third-to-last, etc.)
        """
        super().__init__()
        self.layer_depth = layer_depth

        original_unet = UNet(**unet_params)
        # Create a truncated model
        self.truncated_model = self._create_truncated_model(original_unet)

    def _create_truncated_model(self, original_unet):
        """Creates a model that computes up to the specified layer"""
        if self.layer_depth == -2:
            # For second-to-last layer, we take everything except the final up_path
            return nn.Sequential(
                original_unet.model[0],  # Down path
                original_unet.model[1]  # Skip connection with subblocks
            )
        elif self.layer_depth == -3:
            # For third-to-last layer, we need to modify the skip connection
            # to not include its final component
            down_path = original_unet.model[0]
            skip_connection = original_unet.model[1]

            # Extract the submodule from the skip connection (without its up_path)
            if hasattr(skip_connection, 'submodule'):
                if isinstance(skip_connection.submodule, nn.Sequential) and len(skip_connection.submodule) >= 2:
                    modified_skip = nn.Sequential(
                        skip_connection.submodule[0],
                        skip_connection.submodule[1]
                    )
                    return nn.Sequential(down_path, modified_skip)

            raise ValueError("Cannot create a model for layer_depth=-3 with this UNet architecture")
        else:
            raise ValueError(f"layer_depth={self.layer_depth} is not supported yet")

    def forward(self, x):
        """Forward pass returning the output of the specified intermediate layer"""
        return self.truncated_model(x)



if __name__ == "__main__":
    # Usage example
    unet_params = {
        'spatial_dims': 2,
        'in_channels': 1,
        'out_channels': 32,
        'channels': (8, 16, 32), # 32, 64, 128),
        'strides': (2, 2, 2), # 2, 2, 2)
    }


    # Create the truncated model
    second_last_layer_model = TruncatedUNet(unet_params, layer_depth=-2)
    third_last_layer_model = TruncatedUNet(unet_params, layer_depth=-3)


    # Use the model
    input_tensor = torch.randn(1, 1, 64, 64)
    print(f"Shape of second-to-last layer activation: {second_last_layer_model(input_tensor).shape}")
    print(f"Shape of third-to-last layer activation: {third_last_layer_model(input_tensor).shape}")