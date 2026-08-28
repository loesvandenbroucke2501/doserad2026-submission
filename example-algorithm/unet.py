import torch
import torch.nn as nn
import torch.nn.functional as F

def load_model(model_dir, device):

    model = UNet_3D(
        in_channels = 2, 
        out_channels = 1, 
        nb_filters_start = 16,
        nb_convolutional_blocks = 3,
        activation = 'relu',
        activation_final_layer = 'relu',
        norm = 'instance',
        residual = True)
    #model = model.to(device)

    #state_dict = torch.load(model_dir / 'model.pth', map_location=device)
    #model.load_state_dict(state_dict)

    return model
    
class UNET3D_ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, activation = 'relu', norm = 'batch', residual = False):
        super().__init__()

        mid_channels = out_channels

        self.residual = residual

        def get_normalization_layer(num_channels):

            if norm == 'batch':
                return nn.BatchNorm3d(num_channels)
            elif norm == 'instance':
                return nn.InstanceNorm3d(num_channels)
            elif norm == 'group':
                return nn.GroupNorm(num_groups=4, num_channels=num_channels)
            elif norm == 'none':
                return nn.Identity()
            else:
                raise ValueError(f"Unsupported normalization type: {norm}")
            
        def get_activation_layer():
            if activation == 'relu':
                return nn.ReLU(inplace=True)
            elif activation == 'leaky_relu':
                return nn.LeakyReLU(0.01, inplace=True)
            elif activation == 'prelu':
                return nn.PReLU()
            elif activation == 'none':
                return nn.Identity()
            else:
                raise ValueError(f"Unsupported activation type: {activation}")
            
        
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            get_normalization_layer(mid_channels),
            get_activation_layer(),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            get_normalization_layer(out_channels),
        )

        self.activation = get_activation_layer()

        if in_channels != out_channels:
            self.projection = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.projection = nn.Identity()

    def forward(self, x):
        out = self.conv_block(x)

        if self.residual == True:
            res = self.projection(x)
            out = self.activation(out + res)
        else:
            out = self.activation(out)

        return out

class UNET3D_DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation, norm):
        super().__init__()

        # feature extraction before downsampling
        self.conv = UNET3D_ConvBlock(in_channels, out_channels, activation, norm, residual=False)

        self.down = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.down(x)
        return x

class UNET3D_UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation, norm):
        super().__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv = UNET3D_ConvBlock(out_channels, out_channels, activation, norm, residual=False)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class UNET3D_OutputBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation = 'relu'):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none' or activation == 'linear':
            self.activation = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class UNet_3D(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 nb_filters_start,
                 nb_convolutional_blocks,
                 activation,
                 activation_final_layer,
                 norm,
                 residual = False,
                 dropout_prob = 0.2):
        super(UNet_3D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nb_filters_start = nb_filters_start
        self.nb_convolutional_blocks = nb_convolutional_blocks

        self.encoder = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        filters = nb_filters_start
        for _ in range(nb_convolutional_blocks):
            self.encoder.append(UNET3D_ConvBlock(in_channels, filters, activation, norm, residual))
            self.dropout_layers.append(nn.Dropout3d(p=dropout_prob))
            self.down_blocks.append(UNET3D_DownBlock(filters, filters, activation, norm))
            in_channels = filters
            filters *= 2

        self.bottleneck = UNET3D_ConvBlock(filters // 2, filters, activation, norm, residual)
        self.bottleneck_dropout = nn.Dropout3d(p=dropout_prob)

        self.up_blocks = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for _ in range(nb_convolutional_blocks):
            self.up_blocks.append(UNET3D_UpBlock(filters, filters // 2, activation, norm))
            
            self.decoder.append(UNET3D_ConvBlock(filters, filters // 2, activation, norm, residual))
            filters //= 2

        self.output_block = UNET3D_OutputBlock(filters, out_channels, activation_final_layer)

    def forward(self, x):

        # encoder path
        skip_connections = []
        for encoder, down_block, dropout in zip(self.encoder, self.down_blocks, self.dropout_layers):
            x = encoder(x)
            x = dropout(x)
            skip_connections.append(x)
            x = down_block(x)

        x = self.bottleneck(x)
        x = self.bottleneck_dropout(x)

        for up_block, decoder, skip in zip(self.up_blocks, self.decoder, reversed(skip_connections)):
            
            x = up_block(x)
            x = torch.cat((x, skip), dim=1)
            x = decoder(x)

        return self.output_block(x)