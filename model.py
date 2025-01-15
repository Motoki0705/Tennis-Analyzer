import torch
import torch.nn as nn
import torch.nn.functional as F

class Xception(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Xception, self).__init__()
        
        mid_channels = in_channels // 4
        self.pointwise_in = nn.Conv2d(
            in_channels, mid_channels, kernel_size=1, stride=1, bias=False
            )
        
        self.depthwise = nn.Conv2d(
            mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False
            )
        
        self.pointwise_out = nn.Conv2d(
            mid_channels, out_channels, kernel_size=1, stride=1, bias=False
            )
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        
        self.shortcut = None
        if in_channels!= out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, bias=False
                )
            self.bn2 = nn.BatchNorm2d(out_channels)
            
    def forward(self, x):
        residual = x
        x = self.pointwise_in(x)
        x = self.depthwise(x)
        x = self.pointwise_out(x)
        x = self.bn1(x)
        
        if self.shortcut is not None:
            residual = self.shortcut(residual)
            residual = self.bn2(residual)
            
        x = x + residual
        return self.gelu(x)
    
class ConvGRUCell(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels

        # Gates: update (z) and reset (r)
        self.conv_z = Xception(in_channels + hidden_channels, hidden_channels)
        self.conv_r = Xception(in_channels + hidden_channels, hidden_channels)
        
        # Candidate hidden state
        self.conv_h = Xception(in_channels + hidden_channels, hidden_channels)

    def forward(self, x, h):
        # Concatenate input and hidden state along the channel dimension
        combined = torch.cat([x, h], dim=1)

        # Compute gates
        z = torch.sigmoid(self.conv_z(combined))  # Update gate
        r = torch.sigmoid(self.conv_r(combined))  # Reset gate

        # Compute candidate hidden state
        combined_reset = torch.cat([x, r * h], dim=1)
        h_tilde = torch.tanh(self.conv_h(combined_reset))

        # Compute new hidden state
        h_new = (1 - z) * h + z * h_tilde
        return h_new
    
class ConvGRU(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super().__init__()
        self.num_layers = num_layers
        
        # Create layers of ConvGRUCells
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            input_channels = in_channels if i == 0 else hidden_channels
            self.layers.append(ConvGRUCell(input_channels, hidden_channels))

    def forward(self, x, hidden=None):
        # x: (in_channels, height, width)
        _, height, width = x.size()
        x = torch.unsqueeze(x, dim=0)
        
        if hidden is None:
            # Initialize hidden states with zeros
            hidden = [
                torch.zeros(1, layer.hidden_channels, height, width, device=x.device)
                for layer in self.layers
            ]

        # Process the input sequence
        for i, layer in enumerate(self.layers):
            h_t = hidden[i]
            hidden[i] = layer(x, h_t)  # Update hidden state
            x = hidden[i]  # Pass to the next layer

        return hidden[-1], hidden
    
# Encoder module definition
class Encoder(nn.Module):
    def __init__(self, num_blocks=4, input_channels=1):
        super(Encoder, self).__init__()
        self.num_blocks = num_blocks

        # Convolutional layers for down-sampling
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels=input_channels if i == 0 else 2 ** (5 + i - 1),
                    out_channels=2 ** (5 + i),
                    kernel_size=3, stride=2, padding=1
                ),
                nn.BatchNorm2d(2 ** (5 + i)),
                nn.GELU()
            ) for i in range(num_blocks)
        ])

        # Two Xception blocks
        self.xception_blocks = nn.ModuleList([
            Xception(2 ** (5 + num_blocks - 1), 2 ** (5 + num_blocks - 1)) for _ in range(4)
        ])

    def forward(self, x):
        x_resolutions = []

        # Apply encoder layers
        for idx, layer in enumerate(self.encoder_layers):
            x = layer(x)
            x_resolutions.append(x)

        # Apply Xception blocks
        for block in self.xception_blocks:
            x = block(x)
            
        return x, x_resolutions
    
# Decoder module definition
class Decoder(nn.Module):
    def __init__(self, num_blocks=4):
        super(Decoder, self).__init__()
        self.num_blocks = num_blocks
        self.skip_scale = nn.Parameter(torch.tensor(0.2))
        
        # ConvGRU layer for temporal processing
        self.conv_gru = ConvGRU(
            2 ** (5 + num_blocks - 1),
            2 ** (5 + num_blocks - 1),
            num_layers=3
        )


        self.avgpooling = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc_state = nn.Sequential(
            nn.Linear(2 ** (5 + num_blocks -1), 64),
            nn.InstanceNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.fc_event = nn.Sequential(
            nn.Linear(2 ** (5 + num_blocks - 1), 64),
            nn.InstanceNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 14)
        )
            
        # Transposed convolutional layers for up-sampling
        self.decoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=2 ** (5 + num_blocks - i - 1),
                    out_channels=(2 ** (5 + num_blocks - i - 2)),
                    kernel_size=3, stride=2, padding=1,
                    output_padding=(0, 1) if i == 0 else 1
                ),
                nn.BatchNorm2d((2 ** (5 + num_blocks - i - 2))),
                nn.GELU()
            ) for i in range(num_blocks)
        ])
        
        self.final_conv = nn.Conv2d(2 ** num_blocks, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x, x_resolutions, hidden_states=None, is_state=False, is_event=False, is_track=False):
        _, hidden_states = self.conv_gru(x, hidden_states)  # Apply ConvGRU
        x = hidden_states[-1]  # Use the last hidden state
        
        if is_state:
            x = self.avgpooling(x)
            x = self.flatten(x)
            x_shortcut = self.avgpooling(x_resolutions)
            x_shortcut = self.flatten(x_shortcut)
            x = self.fc_state(x + x_shortcut * self.skip_scale)
            
        elif is_event:
            x = self.avgpooling(x)
            x = self.flatten(x)
            x_shortcut = self.avgpooling(x_resolutions)
            x_shortcut = self.flatten(x_shortcut)
            x = self.fc_event(x + x_shortcut * self.skip_scale)

        elif is_track:
            # Apply decoder layers with skip connections
            for layer, skip_connection in zip(self.decoder_layers, reversed(x_resolutions)):
                x = layer(x + skip_connection * self.skip_scale)  # Skip connection with scaling
            
            x = self.final_conv(x)  # Apply final convolutional layer
            x = x.view(x.size(0), -1)

        return F.softmax(x, dim=1), hidden_states # Flatten and apply softmax 
           
# Example usage
if __name__ == "__main__":
    pass