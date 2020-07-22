import torch.nn as nn
import torch
import torchvision.models

import sys
sys.path.append("..")
from driving_uncertainty.image_dissimilarity.models.semantic_encoder import SemanticEncoder, ResNetSemanticEncoder
from driving_uncertainty.image_dissimilarity.models.vgg_features import VGGFeatures, VGGSPADE
from driving_uncertainty.image_dissimilarity.models.resnet_features import resnet
from driving_uncertainty.image_dissimilarity.models.normalization import SPADE, FILM, GuideCorrelation, GuideNormalization

class DissimNet(nn.Module):
    def __init__(self, architecture='vgg16', semantic=True, pretrained=True, correlation = True, prior = False, spade='',
                 num_semantic_classes = 19):
        super(DissimNet, self).__init__()
        
        #get initialization parameters
        self.correlation = correlation
        self.spade = spade
        self.semantic = semantic

        # generate encoders
        if self.spade == 'encoder' or self.spade == 'both':
            self.vgg_encoder = VGGSPADE(pretrained=pretrained, label_nc=num_semantic_classes)
        else:
            self.vgg_encoder = VGGFeatures(architecture=architecture, pretrained=pretrained)

        if self.semantic:
            self.semantic_encoder = SemanticEncoder(architecture=architecture, in_channels=num_semantic_classes)
        
        # layers for decoder
        # all the 3x3 convolutions
        if correlation:
            self.conv1 = nn.Sequential(nn.Conv2d(513, 256, kernel_size=3, padding=1), nn.SELU())
            self.conv12 = nn.Sequential(nn.Conv2d(513, 256, kernel_size=3, padding=1), nn.SELU())
            self.conv3 = nn.Sequential(nn.Conv2d(385, 128, kernel_size=3, padding=1), nn.SELU())
            self.conv5 = nn.Sequential(nn.Conv2d(193, 64, kernel_size=3, padding=1), nn.SELU())

        else:
            self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.SELU())
            self.conv12 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.SELU())
            self.conv3 = nn.Sequential(nn.Conv2d(384, 128, kernel_size=3, padding=1), nn.SELU())
            self.conv5 = nn.Sequential(nn.Conv2d(192, 64, kernel_size=3, padding=1), nn.SELU())

        if self.spade == 'decoder' or self.spade == 'both':
            self.conv2 = SPADEDecoderLayer(nc=256, label_nc=num_semantic_classes)
            self.conv13 = SPADEDecoderLayer(nc=256, label_nc=num_semantic_classes)
            self.conv4 = SPADEDecoderLayer(nc=128, label_nc=num_semantic_classes)
            self.conv6 = SPADEDecoderLayer(nc=64, label_nc=num_semantic_classes)
        else:
            self.conv2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.SELU())
            self.conv13 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.SELU())
            self.conv4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.SELU())
            self.conv6 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.SELU())

        # all the tranposed convolutions
        self.tconv1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0)
        self.tconv3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0)
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
        if self.spade == 'encoder' or self.spade == 'both':
            encoding_og = self.vgg_encoder(original_img, semantic_img)
            encoding_syn = self.vgg_encoder(synthesis_img, semantic_img)
        else:
            encoding_og = self.vgg_encoder(original_img)
            encoding_syn = self.vgg_encoder(synthesis_img)
        
        if self.semantic:
            encoding_sem = self.semantic_encoder(semantic_img)
            # concatenate the output of each encoder
            layer1_cat = torch.cat((encoding_og[0], encoding_syn[0], encoding_sem[0]), dim=1)
            layer2_cat = torch.cat((encoding_og[1], encoding_syn[1], encoding_sem[1]), dim=1)
            layer3_cat = torch.cat((encoding_og[2], encoding_syn[2], encoding_sem[2]), dim=1)
            layer4_cat = torch.cat((encoding_og[3], encoding_syn[3], encoding_sem[3]), dim=1)
        else:
            layer1_cat = torch.cat((encoding_og[0], encoding_syn[0]), dim=1)
            layer2_cat = torch.cat((encoding_og[1], encoding_syn[1]), dim=1)
            layer3_cat = torch.cat((encoding_og[2], encoding_syn[2]), dim=1)
            layer4_cat = torch.cat((encoding_og[3], encoding_syn[3]), dim=1)
                
        # use 1x1 convolutions to reduce dimensions of concatenations
        layer4_cat = self.conv7(layer4_cat)
        layer3_cat = self.conv8(layer3_cat)
        layer2_cat = self.conv9(layer2_cat)
        layer1_cat = self.conv10(layer1_cat)
        
        if self.correlation:
            # get correlation for each layer (multiplication + 1x1 conv)
            corr1 = torch.sum(torch.mul(encoding_og[0], encoding_syn[0]), dim=1).unsqueeze(dim=1)
            corr2 = torch.sum(torch.mul(encoding_og[1], encoding_syn[1]), dim=1).unsqueeze(dim=1)
            corr3 = torch.sum(torch.mul(encoding_og[2], encoding_syn[2]), dim=1).unsqueeze(dim=1)
            corr4 = torch.sum(torch.mul(encoding_og[3], encoding_syn[3]), dim=1).unsqueeze(dim=1)
            
            # concatenate correlation layers
            layer4_cat = torch.cat((corr4, layer4_cat), dim = 1)
            layer3_cat = torch.cat((corr3, layer3_cat), dim = 1)
            layer2_cat = torch.cat((corr2, layer2_cat), dim = 1)
            layer1_cat = torch.cat((corr1, layer1_cat), dim = 1)

        # Run Decoder
        x = self.conv1(layer4_cat)
        if self.spade == 'decoder' or self.spade == 'both':
            x = self.conv2(x, semantic_img)
        else:
            x = self.conv2(x)
        x = self.tconv1(x)
        
        x = torch.cat((x, layer3_cat), dim=1)
        x = self.conv12(x)
        if self.spade == 'decoder' or self.spade == 'both':
            x = self.conv13(x, semantic_img)
        else:
            x = self.conv13(x)
        x = self.tconv3(x)

        x = torch.cat((x, layer2_cat), dim=1)
        x = self.conv3(x)
        if self.spade == 'decoder' or self.spade == 'both':
            x = self.conv4(x, semantic_img)
        else:
            x = self.conv4(x)
        x = self.tconv2(x)

        x = torch.cat((x, layer1_cat), dim=1)
        x = self.conv5(x)
        if self.spade == 'decoder' or self.spade == 'both':
            x = self.conv6(x, semantic_img)
        else:
            x = self.conv6(x)
        logits = self.conv11(x)
        return logits


class DissimNetPrior(nn.Module):
    def __init__(self, architecture='vgg16', semantic=True, pretrained=True, correlation=True, prior=False, spade='',
                 num_semantic_classes=19):
        super(DissimNetPrior, self).__init__()

        # get initialization parameters
        self.correlation = correlation
        self.spade = spade
        # self.semantic = False if spade else semantic
        self.semantic = semantic
        self.prior = prior

        # generate encoders
        if self.spade == 'encoder' or self.spade == 'both':
            self.vgg_encoder = VGGSPADE(pretrained=pretrained, label_nc=num_semantic_classes)
        else:
            self.vgg_encoder = VGGFeatures(architecture=architecture, pretrained=pretrained)

        if self.semantic:
            self.semantic_encoder = SemanticEncoder(architecture=architecture, in_channels=num_semantic_classes)
            self.prior_encoder = SemanticEncoder(architecture=architecture, in_channels=3, base_feature_size=64)

        # layers for decoder
        # all the 3x3 convolutions
        if correlation:
            self.conv1 = nn.Sequential(nn.Conv2d(513, 256, kernel_size=3, padding=1), nn.SELU())
            self.conv12 = nn.Sequential(nn.Conv2d(513, 256, kernel_size=3, padding=1), nn.SELU())
            self.conv3 = nn.Sequential(nn.Conv2d(385, 128, kernel_size=3, padding=1), nn.SELU())
            self.conv5 = nn.Sequential(nn.Conv2d(193, 64, kernel_size=3, padding=1), nn.SELU())

        else:
            self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.SELU())
            self.conv12 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.SELU())
            self.conv3 = nn.Sequential(nn.Conv2d(384, 128, kernel_size=3, padding=1), nn.SELU())
            self.conv5 = nn.Sequential(nn.Conv2d(192, 64, kernel_size=3, padding=1), nn.SELU())

        if self.spade == 'decoder' or self.spade == 'both':
            self.conv2 = SPADEDecoderLayer(nc=256, label_nc=num_semantic_classes)
            self.conv13 = SPADEDecoderLayer(nc=256, label_nc=num_semantic_classes)
            self.conv4 = SPADEDecoderLayer(nc=128, label_nc=num_semantic_classes)
            self.conv6 = SPADEDecoderLayer(nc=64, label_nc=num_semantic_classes)
        else:
            self.conv2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.SELU())
            self.conv13 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.SELU())
            self.conv4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.SELU())
            self.conv6 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.SELU())

        # all the tranposed convolutions
        self.tconv1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0)
        self.tconv3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0)
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

        # self._initialize_weights()

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

    def forward(self, original_img, synthesis_img, semantic_img, entropy, mae, distance, softmax_out=False):
        # get all the image encodings
        prior_img = torch.cat((entropy, mae, distance), dim=1)
        if self.spade == 'encoder' or self.spade == 'both':
            encoding_og = self.vgg_encoder(original_img, semantic_img)
            encoding_syn = self.vgg_encoder(synthesis_img, semantic_img)
        else:
            encoding_og = self.vgg_encoder(original_img)
            encoding_syn = self.vgg_encoder(synthesis_img)
    
        if self.semantic:
            encoding_sem = self.semantic_encoder(semantic_img)
            # concatenate the output of each encoder
            layer1_cat = torch.cat((encoding_og[0], encoding_syn[0], encoding_sem[0]), dim=1)
            layer2_cat = torch.cat((encoding_og[1], encoding_syn[1], encoding_sem[1]), dim=1)
            layer3_cat = torch.cat((encoding_og[2], encoding_syn[2], encoding_sem[2]), dim=1)
            layer4_cat = torch.cat((encoding_og[3], encoding_syn[3], encoding_sem[3]), dim=1)
        else:
            layer1_cat = torch.cat((encoding_og[0], encoding_syn[0]), dim=1)
            layer2_cat = torch.cat((encoding_og[1], encoding_syn[1]), dim=1)
            layer3_cat = torch.cat((encoding_og[2], encoding_syn[2]), dim=1)
            layer4_cat = torch.cat((encoding_og[3], encoding_syn[3]), dim=1)
    
        # use 1x1 convolutions to reduce dimensions of concatenations
        layer4_cat = self.conv7(layer4_cat)
        layer3_cat = self.conv8(layer3_cat)
        layer2_cat = self.conv9(layer2_cat)
        layer1_cat = self.conv10(layer1_cat)
        
        if self.prior:
            encoding_pior = self.prior_encoder(prior_img)
            layer1_cat = torch.mul(layer1_cat, encoding_pior[0])
            layer2_cat = torch.mul(layer2_cat, encoding_pior[1])
            layer3_cat = torch.mul(layer3_cat, encoding_pior[2])
            layer4_cat = torch.mul(layer4_cat, encoding_pior[3])
    
        if self.correlation:
            # get correlation for each layer (multiplication + 1x1 conv)
            corr1 = torch.sum(torch.mul(encoding_og[0], encoding_syn[0]), dim=1).unsqueeze(dim=1)
            corr2 = torch.sum(torch.mul(encoding_og[1], encoding_syn[1]), dim=1).unsqueeze(dim=1)
            corr3 = torch.sum(torch.mul(encoding_og[2], encoding_syn[2]), dim=1).unsqueeze(dim=1)
            corr4 = torch.sum(torch.mul(encoding_og[3], encoding_syn[3]), dim=1).unsqueeze(dim=1)
        
            # concatenate correlation layers
            layer4_cat = torch.cat((corr4, layer4_cat), dim=1)
            layer3_cat = torch.cat((corr3, layer3_cat), dim=1)
            layer2_cat = torch.cat((corr2, layer2_cat), dim=1)
            layer1_cat = torch.cat((corr1, layer1_cat), dim=1)
    
        # Run Decoder
        x = self.conv1(layer4_cat)
        if self.spade == 'decoder' or self.spade == 'both':
            x = self.conv2(x, semantic_img)
        else:
            x = self.conv2(x)
        x = self.tconv1(x)
    
        x = torch.cat((x, layer3_cat), dim=1)
        x = self.conv12(x)
        if self.spade == 'decoder' or self.spade == 'both':
            x = self.conv13(x, semantic_img)
        else:
            x = self.conv13(x)
        x = self.tconv3(x)
    
        x = torch.cat((x, layer2_cat), dim=1)
        x = self.conv3(x)
        if self.spade == 'decoder' or self.spade == 'both':
            x = self.conv4(x, semantic_img)
        else:
            x = self.conv4(x)
        x = self.tconv2(x)
    
        x = torch.cat((x, layer1_cat), dim=1)
        x = self.conv5(x)
        if self.spade == 'decoder' or self.spade == 'both':
            x = self.conv6(x, semantic_img)
        else:
            x = self.conv6(x)
        logits = self.conv11(x)
        return logits


class ResNetDissimNet(nn.Module):
    def __init__(self, architecture='resnet18', semantic=True, pretrained=True, correlation=True, spade='',
                 num_semantic_classes = 19):
        super(ResNetDissimNet, self).__init__()

        # get initialization parameters
        self.correlation = correlation
        self.spade = spade
        self.semantic = False if spade else semantic
        
        # generate encoders
        if self.spade == 'encoder' or self.spade == 'both':
            raise NotImplementedError()
            #self.encoder = VGGSPADE()
        else:
            self.encoder = resnet(architecture=architecture, pretrained=pretrained)
        
        if self.semantic:
            self.semantic_encoder = ResNetSemanticEncoder()
        
        # layers for decoder
        # all the 3x3 convolutions
        if correlation:
            self.conv1 = nn.Sequential(nn.Conv2d(513, 256, kernel_size=3, padding=1), nn.SELU())
            self.conv12 = nn.Sequential(nn.Conv2d(513, 256, kernel_size=3, padding=1), nn.SELU())
            self.conv3 = nn.Sequential(nn.Conv2d(385, 128, kernel_size=3, padding=1), nn.SELU())
            self.conv5 = nn.Sequential(nn.Conv2d(193, 64, kernel_size=3, padding=1), nn.SELU())

        else:
            self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.SELU())
            self.conv12 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.SELU())
            self.conv3 = nn.Sequential(nn.Conv2d(384, 128, kernel_size=3, padding=1), nn.SELU())
            self.conv5 = nn.Sequential(nn.Conv2d(192, 64, kernel_size=3, padding=1), nn.SELU())
        
        if self.spade == 'decoder' or self.spade == 'both':
            self.conv2 = SPADEDecoderLayer(nc=256, label_nc=num_semantic_classes)
            self.conv13 = SPADEDecoderLayer(nc=256, label_nc=num_semantic_classes)
            self.conv4 = SPADEDecoderLayer(nc=128, label_nc=num_semantic_classes)
            self.conv6 = SPADEDecoderLayer(nc=64, label_nc=num_semantic_classes)
        else:
            self.conv2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.SELU())
            self.conv13 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.SELU())
            self.conv4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.SELU())
            self.conv6 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.SELU())
        
        # all the tranposed convolutions
        self.tconv1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0)
        self.tconv5 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0)
        self.tconv2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0)
        self.tconv3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0)
        self.tconv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0)
        
        # all the other 1x1 convolutions
        if self.semantic:
            self.conv7 = nn.Conv2d(1280, 512, kernel_size=1, padding=0)
            self.conv8 = nn.Conv2d(640, 256, kernel_size=1, padding=0)
            self.conv9 = nn.Conv2d(320, 128, kernel_size=1, padding=0)
            self.conv10 = nn.Conv2d(160, 64, kernel_size=1, padding=0)
            self.conv11 = nn.Conv2d(32, 2, kernel_size=1, padding=0)
        else:
            self.conv7 = nn.Conv2d(1024, 512, kernel_size=1, padding=0)
            self.conv8 = nn.Conv2d(512, 256, kernel_size=1, padding=0)
            self.conv9 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
            self.conv10 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
            self.conv11 = nn.Conv2d(32, 2, kernel_size=1, padding=0)

        # self._initialize_weights()
    
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
        if self.spade == 'encoder' or self.spade == 'both':
            self.encoding_og = self.encoder(original_img, semantic_img)
            self.encoding_syn = self.encoder(synthesis_img, semantic_img)
        else:
            self.encoding_og = self.encoder(original_img)
            self.encoding_syn = self.encoder(synthesis_img)
        
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
            layer4_cat = torch.cat((corr4, layer4_cat), dim=1)
            layer3_cat = torch.cat((corr3, layer3_cat), dim=1)
            layer2_cat = torch.cat((corr2, layer2_cat), dim=1)
            layer1_cat = torch.cat((corr1, layer1_cat), dim=1)
        
        # Run Decoder
        x = self.conv1(layer4_cat)
        if self.spade == 'decoder' or self.spade == 'both':
            x = self.conv2(x, semantic_img)
        else:
            x = self.conv2(x)
        x = self.tconv1(x)
        
        x = torch.cat((x, layer3_cat), dim=1)
        x = self.conv12(x)
        if self.spade == 'decoder' or self.spade == 'both':
            x = self.conv13(x, semantic_img)
        else:
            x = self.conv13(x)
        x = self.tconv5(x)
        
        x = torch.cat((x, layer2_cat), dim=1)
        x = self.conv3(x)
        if self.spade == 'decoder' or self.spade == 'both':
            x = self.conv4(x, semantic_img)
        else:
            x = self.conv4(x)
        x = self.tconv2(x)
        
        x = torch.cat((x, layer1_cat), dim=1)
        x = self.conv5(x)
        if self.spade == 'decoder' or self.spade == 'both':
            x = self.conv6(x, semantic_img)
        else:
            x = self.conv6(x)
        x = self.tconv3(x)
        x = self.tconv4(x)
        
        x = self.conv11(x)
        
        self.final_prediction = x
        
        return self.final_prediction

class GuidedDissimNet(nn.Module):
    def __init__(self, architecture='vgg16', semantic=True, pretrained=True, correlation = True, spade=True,
                 num_semantic_classes = 19):
        super(GuidedDissimNet, self).__init__()
        
        vgg_pretrained_features = torchvision.models.vgg16_bn(pretrained=pretrained).features
        
        # Encoder
        self.norm_layer_1 = FILM(nc=64, guide_nc=64)
        self.norm_layer_2 = FILM(nc=64, guide_nc=64)
        self.norm_layer_3 = FILM(nc=128, guide_nc=128)
        self.norm_layer_4 = FILM(nc=128, guide_nc=128)
        self.norm_layer_5 = FILM(nc=256, guide_nc=256)
        self.norm_layer_6 = FILM(nc=256, guide_nc=256)
        self.norm_layer_7 = FILM(nc=256, guide_nc=256)
        self.norm_layer_8 = FILM(nc=512, guide_nc=512)
        self.norm_layer_9 = FILM(nc=512, guide_nc=512)
        self.norm_layer_10 = FILM(nc=512, guide_nc=512)
        self.norm_layer_11 = FILM(nc=64, guide_nc=64)
        self.norm_layer_12 = FILM(nc=64, guide_nc=64)
        self.norm_layer_13 = FILM(nc=128, guide_nc=128)
        self.norm_layer_14 = FILM(nc=128, guide_nc=128)
        self.norm_layer_15 = FILM(nc=256, guide_nc=256)
        self.norm_layer_16 = FILM(nc=256, guide_nc=256)
        self.norm_layer_17 = FILM(nc=256, guide_nc=256)
        self.norm_layer_18 = FILM(nc=512, guide_nc=512)
        self.norm_layer_19 = FILM(nc=512, guide_nc=512)
        self.norm_layer_20 = FILM(nc=512, guide_nc=512)
        
        # TODO Reformat to make it more efficient/clean code
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.slice6 = nn.Sequential()
        self.slice7 = nn.Sequential()
        self.slice8 = nn.Sequential()
        self.slice9 = nn.Sequential()
        self.slice10 = nn.Sequential()
        self.slice11 = nn.Sequential()
        self.slice12 = nn.Sequential()
        self.slice13 = nn.Sequential()
        self.slice14 = nn.Sequential()
        self.slice15 = nn.Sequential()
        self.slice16 = nn.Sequential()
        self.slice17 = nn.Sequential()
        self.slice18 = nn.Sequential()
        self.slice19 = nn.Sequential()
        self.slice20 = nn.Sequential()
        self.slice21 = nn.Sequential()
        self.slice22 = nn.Sequential()
        self.slice23 = nn.Sequential()
        self.slice24 = nn.Sequential()
        self.slice25 = nn.Sequential()
        self.slice26 = nn.Sequential()
        self.slice27 = nn.Sequential()
        self.slice28 = nn.Sequential()
        
        for x in range(1):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
            self.slice15.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 4):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
            self.slice16.add_module(str(x), vgg_pretrained_features[x])
        for x in range(5, 6):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
            self.slice17.add_module(str(x), vgg_pretrained_features[x])
        for x in range(6, 8):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
            self.slice18.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 11):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
            self.slice19.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 13):
            self.slice6.add_module(str(x), vgg_pretrained_features[x])
            self.slice20.add_module(str(x), vgg_pretrained_features[x])
        for x in range(13, 15):
            self.slice7.add_module(str(x), vgg_pretrained_features[x])
            self.slice21.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 18):
            self.slice8.add_module(str(x), vgg_pretrained_features[x])
            self.slice22.add_module(str(x), vgg_pretrained_features[x])
        for x in range(19, 21):
            self.slice9.add_module(str(x), vgg_pretrained_features[x])
            self.slice23.add_module(str(x), vgg_pretrained_features[x])
        for x in range(22, 23):
            self.slice10.add_module(str(x), vgg_pretrained_features[x])
            self.slice24.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 25):
            self.slice11.add_module(str(x), vgg_pretrained_features[x])
            self.slice25.add_module(str(x), vgg_pretrained_features[x])
        for x in range(26, 28):
            self.slice12.add_module(str(x), vgg_pretrained_features[x])
            self.slice26.add_module(str(x), vgg_pretrained_features[x])
        for x in range(29, 31):
            self.slice13.add_module(str(x), vgg_pretrained_features[x])
            self.slice27.add_module(str(x), vgg_pretrained_features[x])
        for x in range(32, 33):
            self.slice14.add_module(str(x), vgg_pretrained_features[x])
            self.slice28.add_module(str(x), vgg_pretrained_features[x])
        
        # layers for decoder
        # all the 3x3 convolutions
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.SELU())
        self.conv12 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.SELU())
        self.conv3 = nn.Sequential(nn.Conv2d(384, 128, kernel_size=3, padding=1), nn.SELU())
        self.conv5 = nn.Sequential(nn.Conv2d(192, 64, kernel_size=3, padding=1), nn.SELU())
        
        # spade decoder
        self.conv2 = SPADEDecoderLayer(nc=256, label_nc=num_semantic_classes)
        self.conv13 = SPADEDecoderLayer(nc=256, label_nc=num_semantic_classes)
        self.conv4 = SPADEDecoderLayer(nc=128, label_nc=num_semantic_classes)
        self.conv6 = SPADEDecoderLayer(nc=64, label_nc=num_semantic_classes)
        
        # all the tranposed convolutions
        self.tconv1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0)
        self.tconv2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0)
        
        # all the other 1x1 convolutions
        self.conv7 = nn.Conv2d(1024, 512, kernel_size=1, padding=0)
        self.conv8 = nn.Conv2d(512, 256, kernel_size=1, padding=0)
        self.conv9 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv10 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.conv11 = nn.Conv2d(64, 2, kernel_size=1, padding=0)

        # self._initialize_weights()
    
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
    
    def forward(self, original_img, synthesis_img, semantic_img):
        # get all the image encodings
        og_1 = self.slice1(original_img)
        syn_1 = self.slice15(synthesis_img)
        
        og_2 = self.norm_layer_1(og_1, syn_1)
        syn_2 = self.norm_layer_11(og_1, syn_1)
        
        og_1 = self.slice2(og_2)
        syn_1 = self.slice16(syn_2)
        
        layer1_og = self.slice3(self.norm_layer_2(og_1, syn_1))
        layer1_syn = self.slice17(self.norm_layer_12(og_1, syn_1))
        
        og_1 = self.slice4(layer1_og)
        syn_1 = self.slice18(layer1_syn)
        
        og_2 = self.norm_layer_3(og_1, syn_1)
        syn_2 = self.norm_layer_13(og_1, syn_1)
        
        og_1 = self.slice5(og_2)
        syn_1 = self.slice19(syn_2)
        
        layer2_og = self.slice6(self.norm_layer_4(og_1, syn_1))
        layer2_syn = self.slice20(self.norm_layer_14(og_1, syn_1))
        
        og_1 = self.slice7(layer2_og)
        syn_1 = self.slice21(layer2_syn)
        
        og_2 = self.norm_layer_5(og_1, syn_1)
        syn_2 = self.norm_layer_15(og_1, syn_1)
        
        og_1 = self.slice8(og_2)
        syn_1 = self.slice22(syn_2)
        
        og_2 = self.norm_layer_6(og_1, syn_1)
        syn_2 = self.norm_layer_16(og_1, syn_1)
        
        og_1 = self.slice9(og_2)
        syn_1 = self.slice23(syn_2)
        
        layer3_og = self.slice10(self.norm_layer_7(og_1, syn_1))
        layer3_syn = self.slice24(self.norm_layer_17(og_1, syn_1))
        
        og_1 = self.slice11(layer3_og)
        syn_1 = self.slice25(layer3_syn)
        
        og_2 = self.norm_layer_8(og_1, syn_1)
        syn_2 = self.norm_layer_18(og_1, syn_1)
        
        og_1 = self.slice12(og_2)
        syn_1 = self.slice26(syn_2)
        
        og_2 = self.norm_layer_9(og_1, syn_1)
        syn_2 = self.norm_layer_19(og_1, syn_1)
        
        og_1 = self.slice13(og_2)
        syn_1 = self.slice27(syn_2)
        
        layer4_og = self.slice14(self.norm_layer_10(og_1, syn_1))
        layer4_syn = self.slice28(self.norm_layer_20(og_1, syn_1))
        
        # concatenate the output of each encoder
        layer1_cat = torch.cat((layer1_og, layer1_syn), dim=1)
        layer2_cat = torch.cat((layer2_og, layer2_syn), dim=1)
        layer3_cat = torch.cat((layer3_og, layer3_syn), dim=1)
        layer4_cat = torch.cat((layer4_og, layer4_syn), dim=1)
        
        # use 1x1 convolutions to reduce dimensions of concatenations
        layer4_cat = self.conv7(layer4_cat)
        layer3_cat = self.conv8(layer3_cat)
        layer2_cat = self.conv9(layer2_cat)
        layer1_cat = self.conv10(layer1_cat)
        
        # Run Decoder
        x = self.conv1(layer4_cat)
        x = self.conv2(x, semantic_img)
        x = self.tconv1(x)
        
        x = torch.cat((x, layer3_cat), dim=1)
        x = self.conv12(x)
        x = self.conv13(x, semantic_img)
        x = self.tconv1(x)
        
        x = torch.cat((x, layer2_cat), dim=1)
        x = self.conv3(x)
        x = self.conv4(x, semantic_img)
        x = self.tconv2(x)
        
        x = torch.cat((x, layer1_cat), dim=1)
        x = self.conv5(x)
        x = self.conv6(x, semantic_img)
        x = self.conv11(x)
        
        self.final_prediction = x
        
        return self.final_prediction

class CorrelatedDissimNet(nn.Module):
    def __init__(self, architecture='vgg16', semantic=True, pretrained=True, correlation=True, spade=True,
                 num_semantic_classes = 19):
        super(CorrelatedDissimNet, self).__init__()

        self.spade = spade
        
        # layers for encoder
        self.og_gel1 = GuideEncoderLayer(nc_in=3, nc_out=64)
        self.syn_gel1 = GuideEncoderLayer(nc_in=3, nc_out=64)
        
        self.og_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.syn_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.og_gc1 = GuideCorrelation(nc=64, guide_nc=64)
        self.og_gc2 = GuideCorrelation(nc=64, guide_nc=num_semantic_classes)
        self.og_gn1 = GuideNormalization(nc=64)
        self.og_relu1 = nn.ReLU(inplace=True)
        self.syn_gc1 = GuideCorrelation(nc=64, guide_nc=64)
        self.syn_gc2 = GuideCorrelation(nc=64, guide_nc=num_semantic_classes)
        self.syn_gn1 = GuideNormalization(nc=64)
        self.syn_relu1 = nn.ReLU(inplace=True)
        self.og_max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.syn_max1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.og_gel2 = GuideEncoderLayer(nc_in=64, nc_out=128)
        self.syn_gel2 = GuideEncoderLayer(nc_in=64, nc_out=128)

        self.og_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.syn_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.og_gc3 = GuideCorrelation(nc=128, guide_nc=128)
        self.og_gc4 = GuideCorrelation(nc=128, guide_nc=num_semantic_classes)
        self.og_gn2 = GuideNormalization(nc=128)
        self.og_relu2 = nn.ReLU(inplace=True)
        self.syn_gc3 = GuideCorrelation(nc=128, guide_nc=128)
        self.syn_gc4 = GuideCorrelation(nc=128, guide_nc=num_semantic_classes)
        self.syn_gn2 = GuideNormalization(nc=128)
        self.syn_relu2 = nn.ReLU(inplace=True)
        self.og_max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.syn_max2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.og_gel3 = GuideEncoderLayer(nc_in=128, nc_out=256)
        self.syn_gel3 = GuideEncoderLayer(nc_in=128, nc_out=256)
        self.og_gel4 = GuideEncoderLayer(nc_in=256, nc_out=256)
        self.syn_gel4 = GuideEncoderLayer(nc_in=256, nc_out=256)

        self.og_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.syn_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.og_gc5 = GuideCorrelation(nc=256, guide_nc=256)
        self.og_gc6 = GuideCorrelation(nc=256, guide_nc=num_semantic_classes)
        self.og_gn3 = GuideNormalization(nc=256)
        self.og_relu3 = nn.ReLU(inplace=True)
        self.syn_gc5 = GuideCorrelation(nc=256, guide_nc=256)
        self.syn_gc6 = GuideCorrelation(nc=256, guide_nc=num_semantic_classes)
        self.syn_gn3 = GuideNormalization(nc=256)
        self.syn_relu3 = nn.ReLU(inplace=True)
        self.og_max3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.syn_max3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.og_gel5 = GuideEncoderLayer(nc_in=256, nc_out=512)
        self.syn_gel5 = GuideEncoderLayer(nc_in=256, nc_out=512)
        self.og_gel6 = GuideEncoderLayer(nc_in=512, nc_out=512)
        self.syn_gel6 = GuideEncoderLayer(nc_in=512, nc_out=512)

        self.og_conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.syn_conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.og_gc7 = GuideCorrelation(nc=512, guide_nc=512)
        self.og_gc8 = GuideCorrelation(nc=512, guide_nc=num_semantic_classes)
        self.og_gn4 = GuideNormalization(nc=512)
        self.og_relu4 = nn.ReLU(inplace=True)
        self.syn_gc7 = GuideCorrelation(nc=512, guide_nc=512)
        self.syn_gc8 = GuideCorrelation(nc=512, guide_nc=num_semantic_classes)
        self.syn_gn4 = GuideNormalization(nc=512)
        self.syn_relu4 = nn.ReLU(inplace=True)
        
        # layers for decoder
        # all the 3x3 convolutions
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.SELU())
        self.conv12 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.SELU())
        self.conv3 = nn.Sequential(nn.Conv2d(384, 128, kernel_size=3, padding=1), nn.SELU())
        self.conv5 = nn.Sequential(nn.Conv2d(192, 64, kernel_size=3, padding=1), nn.SELU())
        
        # spade decoder
        if self.spade == 'decoder' or self.spade == 'both':
            self.conv2 = SPADEDecoderLayer(nc=256, label_nc=num_semantic_classes)
            self.conv13 = SPADEDecoderLayer(nc=256, label_nc=num_semantic_classes)
            self.conv4 = SPADEDecoderLayer(nc=128, label_nc=num_semantic_classes)
            self.conv6 = SPADEDecoderLayer(nc=64, label_nc=num_semantic_classes)
        else:
            self.conv2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.SELU())
            self.conv13 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.SELU())
            self.conv4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.SELU())
            self.conv6 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.SELU())
        
        # all the tranposed convolutions
        self.tconv1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0)
        self.tconv3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0)
        self.tconv2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0)
        
        # all the other 1x1 convolutions
        self.conv7 = nn.Conv2d(1024, 512, kernel_size=1, padding=0)
        self.conv8 = nn.Conv2d(512, 256, kernel_size=1, padding=0)
        self.conv9 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv10 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.conv11 = nn.Conv2d(64, 2, kernel_size=1, padding=0)
        
        # self._initialize_weights()
    
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
    
    def forward(self, original_img, synthesis_img, semantic_img):
        # get all the image encodings
        og = self.og_gel1(original_img)
        syn = self.syn_gel1(synthesis_img)

        og = self.og_conv1(og)
        syn = self.syn_conv1(syn)
        
        gamma1, beta1 = self.og_gc1(og, syn)
        gamma2, beta2 = self.og_gc2(og, semantic_img)

        gamma3, beta3 = self.syn_gc1(syn, og)
        gamma4, beta4 = self.syn_gc2(syn, semantic_img)
        
        layer1_og = self.og_relu1(self.og_gn1(og, gamma1, beta1, gamma2, beta2))
        layer1_syn = self.syn_relu1(self.syn_gn1(syn, gamma3, beta3, gamma4, beta4))

        og = self.og_gel2(self.og_max1(layer1_og))
        syn = self.syn_gel2(self.syn_max1(layer1_syn))

        og = self.og_conv2(og)
        syn = self.syn_conv2(syn)

        gamma1, beta1 = self.og_gc3(og, syn)
        gamma2, beta2 = self.og_gc4(og, semantic_img)

        gamma3, beta3 = self.syn_gc3(syn, og)
        gamma4, beta4 = self.syn_gc4(syn, semantic_img)

        layer2_og = self.og_relu2(self.og_gn2(og, gamma1, beta1, gamma2, beta2))
        layer2_syn = self.syn_relu2(self.syn_gn2(syn, gamma3, beta3, gamma4, beta4))

        og = self.og_gel3(self.og_max2(layer2_og))
        syn = self.syn_gel3(self.syn_max2(layer2_syn))
        og = self.og_gel4(og)
        syn = self.syn_gel4(syn)

        og = self.og_conv3(og)
        syn = self.syn_conv3(syn)

        gamma1, beta1 = self.og_gc5(og, syn)
        gamma2, beta2 = self.og_gc6(og, semantic_img)

        gamma3, beta3 = self.syn_gc5(syn, og)
        gamma4, beta4 = self.syn_gc6(syn, semantic_img)

        layer3_og = self.og_relu3(self.og_gn3(og, gamma1, beta1, gamma2, beta2))
        layer3_syn = self.syn_relu3(self.syn_gn3(syn, gamma3, beta3, gamma4, beta4))
        
        og = self.og_gel5(self.og_max3(layer3_og))
        syn = self.syn_gel5(self.syn_max3(layer3_syn))
        og = self.og_gel6(og)
        syn = self.syn_gel6(syn)

        og = self.og_conv4(og)
        syn = self.syn_conv4(syn)

        gamma1, beta1 = self.og_gc7(og, syn)
        gamma2, beta2 = self.og_gc8(og, semantic_img)

        gamma3, beta3 = self.syn_gc7(syn, og)
        gamma4, beta4 = self.syn_gc8(syn, semantic_img)

        layer4_og = self.og_relu4(self.og_gn4(og, gamma1, beta1, gamma2, beta2))
        layer4_syn = self.syn_relu4(self.syn_gn4(syn, gamma3, beta3, gamma4, beta4))
        
        # concatenate the output of each encoder
        layer1_cat = torch.cat((layer1_og, layer1_syn), dim=1)
        layer2_cat = torch.cat((layer2_og, layer2_syn), dim=1)
        layer3_cat = torch.cat((layer3_og, layer3_syn), dim=1)
        layer4_cat = torch.cat((layer4_og, layer4_syn), dim=1)
        
        # use 1x1 convolutions to reduce dimensions of concatenations
        layer4_cat = self.conv7(layer4_cat)
        layer3_cat = self.conv8(layer3_cat)
        layer2_cat = self.conv9(layer2_cat)
        layer1_cat = self.conv10(layer1_cat)
        
        # Run Decoder
        x = self.conv1(layer4_cat)
        if self.spade == 'decoder' or self.spade == 'both':
            x = self.conv2(x, semantic_img)
        else:
            x = self.conv2(x)
        x = self.tconv1(x)

        x = torch.cat((x, layer3_cat), dim=1)
        x = self.conv12(x)
        if self.spade == 'decoder' or self.spade == 'both':
            x = self.conv13(x, semantic_img)
        else:
            x = self.conv13(x)
        x = self.tconv3(x)

        x = torch.cat((x, layer2_cat), dim=1)
        x = self.conv3(x)
        if self.spade == 'decoder' or self.spade == 'both':
            x = self.conv4(x, semantic_img)
        else:
            x = self.conv4(x)
        x = self.tconv2(x)

        x = torch.cat((x, layer1_cat), dim=1)
        x = self.conv5(x)
        if self.spade == 'decoder' or self.spade == 'both':
            x = self.conv6(x, semantic_img)
        else:
            x = self.conv6(x)
        x = self.conv11(x)
        
        self.final_prediction = x
        
        return self.final_prediction

class CorrelatedDissimNetGuide(nn.Module):
    def __init__(self, architecture='vgg16', semantic=True, pretrained=True, correlation=True, spade='decoder',
                 num_semantic_classes=19):
        super(CorrelatedDissimNetGuide, self).__init__()

        self.spade = spade

        # layers for encoder
        self.og_gel1 = GuideEncoderLayer(nc_in=3, nc_out=64)
        self.syn_gel1 = GuideEncoderLayer(nc_in=3, nc_out=64)

        self.og_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.syn_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.og_gc1 = SPADE(norm_nc=64, label_nc=64)
        self.og_relu1 = nn.ReLU(inplace=True)
        self.syn_gc1 = SPADE(norm_nc=64, label_nc=64)
        self.syn_relu1 = nn.ReLU(inplace=True)
        self.og_max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.syn_max1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.og_gel2 = GuideEncoderLayer(nc_in=64, nc_out=128)
        self.syn_gel2 = GuideEncoderLayer(nc_in=64, nc_out=128)

        self.og_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.syn_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.og_gc2 = SPADE(norm_nc=128, label_nc=128)
        self.og_relu2 = nn.ReLU(inplace=True)
        self.syn_gc2 = SPADE(norm_nc=128, label_nc=128)
        self.syn_relu2 = nn.ReLU(inplace=True)
        self.og_max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.syn_max2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.og_gel3 = GuideEncoderLayer(nc_in=128, nc_out=256)
        self.syn_gel3 = GuideEncoderLayer(nc_in=128, nc_out=256)
        self.og_gel4 = GuideEncoderLayer(nc_in=256, nc_out=256)
        self.syn_gel4 = GuideEncoderLayer(nc_in=256, nc_out=256)

        self.og_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.syn_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.og_gc3 = SPADE(norm_nc=256, label_nc=256)
        self.og_relu3 = nn.ReLU(inplace=True)
        self.syn_gc3 = SPADE(norm_nc=256, label_nc=256)
        self.syn_relu3 = nn.ReLU(inplace=True)
        self.og_max3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.syn_max3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.og_gel5 = GuideEncoderLayer(nc_in=256, nc_out=512)
        self.syn_gel5 = GuideEncoderLayer(nc_in=256, nc_out=512)
        self.og_gel6 = GuideEncoderLayer(nc_in=512, nc_out=512)
        self.syn_gel6 = GuideEncoderLayer(nc_in=512, nc_out=512)

        self.og_conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.syn_conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.og_gc4 = SPADE(norm_nc=512, label_nc=512)
        self.og_relu4 = nn.ReLU(inplace=True)
        self.syn_gc4 = SPADE(norm_nc=512, label_nc=512)
        self.syn_relu4 = nn.ReLU(inplace=True)

        # layers for decoder
        # all the 3x3 convolutions
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.SELU())
        self.conv12 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.SELU())
        self.conv3 = nn.Sequential(nn.Conv2d(384, 128, kernel_size=3, padding=1), nn.SELU())
        self.conv5 = nn.Sequential(nn.Conv2d(192, 64, kernel_size=3, padding=1), nn.SELU())

        # spade decoder
        if self.spade == 'decoder' or self.spade == 'both':
            self.conv2 = SPADEDecoderLayer(nc=256, label_nc=num_semantic_classes)
            self.conv13 = SPADEDecoderLayer(nc=256, label_nc=num_semantic_classes)
            self.conv4 = SPADEDecoderLayer(nc=128, label_nc=num_semantic_classes)
            self.conv6 = SPADEDecoderLayer(nc=64, label_nc=num_semantic_classes)
        else:
            self.conv2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.SELU())
            self.conv13 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.SELU())
            self.conv4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.SELU())
            self.conv6 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.SELU())

        # all the tranposed convolutions
        self.tconv1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0)
        self.tconv3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0)
        self.tconv2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0)

        # all the other 1x1 convolutions
        self.conv7 = nn.Conv2d(1024, 512, kernel_size=1, padding=0)
        self.conv8 = nn.Conv2d(512, 256, kernel_size=1, padding=0)
        self.conv9 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv10 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.conv11 = nn.Conv2d(64, 2, kernel_size=1, padding=0)

        # self._initialize_weights()

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

    def forward(self, original_img, synthesis_img, semantic_img):
        # get all the image encodings
        og = self.og_gel1(original_img)
        syn = self.syn_gel1(synthesis_img)

        og_1 = self.og_conv1(og)
        syn_1 = self.syn_conv1(syn)

        og_2 = self.og_gc1(og_1, syn_1)
        syn_2 = self.syn_gc1(syn_1, og_1)

        layer1_og = self.og_relu1(og_2)
        layer1_syn = self.syn_relu1(syn_2)

        og = self.og_gel2(self.og_max1(layer1_og))
        syn = self.syn_gel2(self.syn_max1(layer1_syn))

        og_1 = self.og_conv2(og)
        syn_1 = self.syn_conv2(syn)

        og_2 = self.og_gc2(og_1, syn_1)
        syn_2 = self.syn_gc2(syn_1, og_1)

        layer2_og = self.og_relu2(og_2)
        layer2_syn = self.syn_relu2(syn_2)

        og = self.og_gel3(self.og_max2(layer2_og))
        syn = self.syn_gel3(self.syn_max2(layer2_syn))
        og = self.og_gel4(og)
        syn = self.syn_gel4(syn)

        og_1 = self.og_conv3(og)
        syn_1 = self.syn_conv3(syn)

        og_2 = self.og_gc3(og_1, syn_1)
        syn_2 = self.syn_gc3(syn_1, og_1)

        layer3_og = self.og_relu3(og_2)
        layer3_syn = self.syn_relu3(syn_2)

        og = self.og_gel5(self.og_max3(layer3_og))
        syn = self.syn_gel5(self.syn_max3(layer3_syn))
        og = self.og_gel6(og)
        syn = self.syn_gel6(syn)

        og_1 = self.og_conv4(og)
        syn_1 = self.syn_conv4(syn)

        og_2 = self.og_gc4(og_1, syn_1)
        syn_2 = self.syn_gc4(syn_1, og_1)

        layer4_og = self.og_relu4(og_2)
        layer4_syn = self.syn_relu4(syn_2)

        # concatenate the output of each encoder
        layer1_cat = torch.cat((layer1_og, layer1_syn), dim=1)
        layer2_cat = torch.cat((layer2_og, layer2_syn), dim=1)
        layer3_cat = torch.cat((layer3_og, layer3_syn), dim=1)
        layer4_cat = torch.cat((layer4_og, layer4_syn), dim=1)

        # use 1x1 convolutions to reduce dimensions of concatenations
        layer4_cat = self.conv7(layer4_cat)
        layer3_cat = self.conv8(layer3_cat)
        layer2_cat = self.conv9(layer2_cat)
        layer1_cat = self.conv10(layer1_cat)

        # Run Decoder
        x = self.conv1(layer4_cat)
        if self.spade == 'decoder' or self.spade == 'both':
            x = self.conv2(x, semantic_img)
        else:
            x = self.conv2(x)
        x = self.tconv1(x)

        x = torch.cat((x, layer3_cat), dim=1)
        x = self.conv12(x)
        if self.spade == 'decoder' or self.spade == 'both':
            x = self.conv13(x, semantic_img)
        else:
            x = self.conv13(x)
        x = self.tconv3(x)

        x = torch.cat((x, layer2_cat), dim=1)
        x = self.conv3(x)
        if self.spade == 'decoder' or self.spade == 'both':
            x = self.conv4(x, semantic_img)
        else:
            x = self.conv4(x)
        x = self.tconv2(x)

        x = torch.cat((x, layer1_cat), dim=1)
        x = self.conv5(x)
        if self.spade == 'decoder' or self.spade == 'both':
            x = self.conv6(x, semantic_img)
        else:
            x = self.conv6(x)
        x = self.conv11(x)

        self.final_prediction = x

        return self.final_prediction

class SPADEDecoderLayer(nn.Module):
    def __init__(self, nc=256, label_nc=19):
        super(SPADEDecoderLayer, self).__init__()

        # create conv layers
        self.norm1 = SPADE(norm_nc=nc, label_nc=label_nc)
        self.selu1 = nn.SELU()
        self.conv = nn.Conv2d(nc, nc, kernel_size=3, padding=1)
        self.norm2 = SPADE(norm_nc=nc, label_nc=label_nc)
        self.selu2 = nn.SELU()

    def forward(self, x, seg):
        out = self.selu2(self.norm2(self.conv(self.selu1(self.norm1(x, seg))), seg))
        return out
    
class GuideEncoderLayer(nn.Module):
    def __init__(self, nc_in=3, nc_out=64):
        super(GuideEncoderLayer, self).__init__()

        # create conv layers
        self.conv = nn.Conv2d(nc_in, nc_out, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(nc_out, affine=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


if __name__ == "__main__":
    from PIL import Image
    import torchvision.models as models
    import torchvision.transforms as transforms
    
    img = Image.open('../../sample_images/zm0002_100000.png')
    diss_model = CorrelatedDissimNet()
    img_transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = img_transform(img)
    outputs = diss_model(img_tensor.unsqueeze(0), img_tensor.unsqueeze(0), img_tensor.unsqueeze(0))
    print(img_tensor[0].data.shape)
    print(outputs.data.shape)
