import torch.nn as nn
import torch

from models.semantic_encoder import SemanticEncoder
from models.vgg_features import VGGFeatures

class DissimNet(nn.Module):
    def __init__(self, architecture='vgg16', semantic=True, pretrained=True, correlation = True):
        super(DissimNet, self).__init__()
        
        self.correlation = correlation
        self.semantic = semantic
        
        # generate encoders
        self.vgg_encoder = VGGFeatures(architecture=architecture, pretrained=pretrained)
        if self.semantic:
            self.semantic_encoder = SemanticEncoder(architecture=architecture)
        
        # layers for decoder
        # all the 3x3 convolutions
        if correlation:
            self.conv1 = nn.Sequential(nn.Conv2d(513, 256, kernel_size=3, padding=1), nn.SELU())
            self.conv12 = nn.Sequential(nn.Conv2d(513, 256, kernel_size=3, padding=1), nn.SELU())
            self.conv3 = nn.Sequential(nn.Conv2d(385, 128, kernel_size=3, padding=1), nn.SELU())
            self.conv5 = nn.Sequential(nn.Conv2d(193, 64, kernel_size=3, padding=1), nn.SELU())
            
            # all correlations 1x1
            self.corr1 = nn.Conv2d(512, 1, kernel_size=1, padding=0)
            self.corr2 = nn.Conv2d(256, 1, kernel_size=1, padding=0)
            self.corr3 = nn.Conv2d(128, 1, kernel_size=1, padding=0)
            self.corr4 = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.SELU())
            self.conv12 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.SELU())
            self.conv3 = nn.Sequential(nn.Conv2d(384, 128, kernel_size=3, padding=1), nn.SELU())
            self.conv5 = nn.Sequential(nn.Conv2d(192, 64, kernel_size=3, padding=1), nn.SELU())
            
        self.conv2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.SELU())
        self.conv13 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.SELU())
        self.conv4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.SELU())
        self.conv6 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.SELU())

        # all the tranposed convolutions
        self.tconv1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0)
        self.tconv2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0)

        # all the other 1x1 convolutions
        if self.semantic:
            self.conv7 = nn.Conv2d(1280, 512, kernel_size=1, padding=0)
            self.conv8 = nn.Conv2d(640, 256, kernel_size=1, padding=0)
            self.conv9 = nn.Conv2d(320, 128, kernel_size=1, padding=0)
            self.conv10 = nn.Conv2d(160, 64, kernel_size=1, padding=0)
            self.conv11 = nn.Conv2d(64, 2, kernel_size=1, padding=0)
        else:
            self.conv7 = nn.Conv2d(1024, 512, kernel_size=1, padding=0)
            self.conv8 = nn.Conv2d(512, 256, kernel_size=1, padding=0)
            self.conv9 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
            self.conv10 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
            self.conv11 = nn.Conv2d(64, 2, kernel_size=1, padding=0)
        
        #self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        

    def forward(self, original_img, synthesis_img, semantic_img, softmax_out=False):
        # get all the image encodings
        self.encoding_og = self.vgg_encoder(original_img)
        self.encoding_syn = self.vgg_encoder(synthesis_img)
        if self.semantic:
            self.encoding_sem = self.semantic_encoder(semantic_img)
            # concatenate the output of each encoder
            layer1_cat = torch.cat((self.encoding_og[0], self.encoding_syn[0], self.encoding_sem[0]), dim=1)
            layer2_cat = torch.cat((self.encoding_og[1], self.encoding_syn[1], self.encoding_sem[1]), dim=1)
            layer3_cat = torch.cat((self.encoding_og[2], self.encoding_syn[2], self.encoding_sem[2]), dim=1)
            layer4_cat = torch.cat((self.encoding_og[3], self.encoding_syn[3], self.encoding_sem[3]), dim=1)
        else:
            layer1_cat = torch.cat((self.encoding_og[0], self.encoding_syn[0]), dim=1)
            layer2_cat = torch.cat((self.encoding_og[1], self.encoding_syn[1]), dim=1)
            layer3_cat = torch.cat((self.encoding_og[2], self.encoding_syn[2]), dim=1)
            layer4_cat = torch.cat((self.encoding_og[3], self.encoding_syn[3]), dim=1)
                
        # use 1x1 convolutions to reduce dimensions of concatenations
        layer4_cat = self.conv7(layer4_cat)
        layer3_cat = self.conv8(layer3_cat)
        layer2_cat = self.conv9(layer2_cat)
        layer1_cat = self.conv10(layer1_cat)
        
        if self.correlation:
            # get correlation for each layer (multiplication + 1x1 conv)
            corr1 = torch.sum(torch.mul(self.encoding_og[0], self.encoding_syn[0]), dim=1).unsqueeze(dim=1)
            corr2 = torch.sum(torch.mul(self.encoding_og[1], self.encoding_syn[1]), dim=1).unsqueeze(dim=1)
            corr3 = torch.sum(torch.mul(self.encoding_og[2], self.encoding_syn[2]), dim=1).unsqueeze(dim=1)
            corr4 = torch.sum(torch.mul(self.encoding_og[3], self.encoding_syn[3]), dim=1).unsqueeze(dim=1)
            
            # concatenate correlation layers
            layer4_cat = torch.cat((corr4, layer4_cat), dim = 1)
            layer3_cat = torch.cat((corr3, layer3_cat), dim = 1)
            layer2_cat = torch.cat((corr2, layer2_cat), dim = 1)
            layer1_cat = torch.cat((corr1, layer1_cat), dim = 1)

        # Run Decoder
        x = self.conv1(layer4_cat)
        x = self.conv2(x)
        x = self.tconv1(x)
        
        x = torch.cat((x, layer3_cat), dim=1)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.tconv1(x)

        x = torch.cat((x, layer2_cat), dim=1)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.tconv2(x)

        x = torch.cat((x, layer1_cat), dim=1)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv11(x)
        
        self.final_prediction = x

        return self.final_prediction

if __name__ == "__main__":
    from PIL import Image
    import torchvision.models as models
    import torchvision.transforms as transforms
    
    img = Image.open('../../sample_images/zm0002_100000.png')
    diss_model = DissimNet()
    img_transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = img_transform(img)
    outputs = diss_model(img_tensor.unsqueeze(0), img_tensor.unsqueeze(0), img_tensor.unsqueeze(0))
    print(img_tensor[0].data.shape)
    print(outputs.data.shape)
