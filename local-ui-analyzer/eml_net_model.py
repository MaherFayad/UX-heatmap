import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class EMLNet(nn.Module):
    """
    EML-NET Architecture (Single Encoder Version).
    
    Based on https://github.com/SenJia/EML-NET-Saliency/blob/master/resnet.py
    
    Uses ResNet50 backbone with multi-scale readout layers and a final combination layer.
    """
    def __init__(self):
        super(EMLNet, self).__init__()
        
        # Load backbone (ResNet50)
        # We use standard torchvision ResNet50 and hook into its layers
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Encoder layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Readout layers (simple decoders)
        # Matches official implementation: Conv3x3 -> BN -> ReLU (or Sigmoid for combined)
        self.output0 = self._make_output(64, readout=1)
        self.output1 = self._make_output(256, readout=1)
        self.output2 = self._make_output(512, readout=1)
        self.output3 = self._make_output(1024, readout=1)
        self.output4 = self._make_output(2048, readout=1)
        
        # Final combination layer
        # Combines the 5 scale outputs
        self.combined = self._make_output(5, readout=1, sigmoid=False) 
        # Note: Official repo uses sigmoid=True in 'combined' but then applies simple sum or another sigmoid?
        # In repo: self.combined = self._make_output(5, sigmoid=True)
        # But we usually want raw logits for loss or sigmoid at the very end.
        # Train script applies sigmoid on output. So we can keep this linear or sigmoid.
        # Official repo 'combined' ends with Sigmoid. 
        # But commonly we need raw logits for numerical stability in some losses, 
        # though CC loss typically handles [0,1].
        # Let's match official: Sigmoid at the end of combined.
        
    def _make_output(self, planes, readout=1, sigmoid=False):
        layers = [
            nn.Conv2d(planes, readout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(readout),
        ]
        if sigmoid:
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        input_size = x.size()[2:]
        
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c0 = x # 64 channels
        
        x = self.maxpool(x)
        c1 = self.layer1(x) # 256 channels
        c2 = self.layer2(c1) # 512 channels
        c3 = self.layer3(c2) # 1024 channels
        c4 = self.layer4(c3) # 2048 channels
        
        # Readout (Multi-scale)
        r0 = self.output0(c0)
        r1 = self.output1(c1)
        r2 = self.output2(c2)
        r3 = self.output3(c3)
        r4 = self.output4(c4)
        
        # Upsample to input size
        r0_up = F.interpolate(r0, size=input_size, mode='bilinear', align_corners=True)
        r1_up = F.interpolate(r1, size=input_size, mode='bilinear', align_corners=True)
        r2_up = F.interpolate(r2, size=input_size, mode='bilinear', align_corners=True)
        r3_up = F.interpolate(r3, size=input_size, mode='bilinear', align_corners=True)
        r4_up = F.interpolate(r4, size=input_size, mode='bilinear', align_corners=True)
        
        # Concatenate
        concat = torch.cat((r0_up, r1_up, r2_up, r3_up, r4_up), dim=1) # (B, 5, H, W)
        
        # Final combination
        out = self.combined(concat)
        
        return out
