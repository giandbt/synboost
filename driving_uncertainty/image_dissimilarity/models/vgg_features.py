import torch.nn as nn
import torchvision.models
import torch
import sys
from torch.nn.modules.upsampling import Upsample
sys.path.append("..")
from driving_uncertainty.image_dissimilarity.models.normalization import SPADE

class VGGFeatures(nn.Module):

	def __init__(self, architecture='vgg16', pretrained=True):
		super(VGGFeatures, self).__init__()
		
		if 'bn' not in architecture:
			vgg16 = torchvision.models.vgg16(pretrained=pretrained)
			
			self.layer1 = nn.Sequential(*list(vgg16.children())[0][:4])
			self.layer2 = nn.Sequential(*list(vgg16.children())[0][4:9])
			self.layer3 = nn.Sequential(*list(vgg16.children())[0][9:16])
			self.layer4 = nn.Sequential(*list(vgg16.children())[0][16:23])
		else:
			vgg16_bn = torchvision.models.vgg16_bn(pretrained=pretrained)
			self.layer1 = nn.Sequential(*list(vgg16_bn.children())[0][:6])
			self.layer2 = nn.Sequential(*list(vgg16_bn.children())[0][6:13])
			self.layer3 = nn.Sequential(*list(vgg16_bn.children())[0][13:23])
			self.layer4 = nn.Sequential(*list(vgg16_bn.children())[0][23:33])
			
	def forward(self, x):
		x1 = self.layer1(x)
		x2 = self.layer2(x1)
		x3 = self.layer3(x2)
		x4 = self.layer4(x3)
		return [x1, x2, x3, x4]


def make_layers(cfg, batch_norm=False):
	layers = []
	in_channels = 3
	for v in cfg:
		if v == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
			in_channels = v
	return nn.Sequential(*layers)


class VGGSPADE(torch.nn.Module):
	def __init__(self, pretrained=True, label_nc=19):
		
		super(VGGSPADE, self).__init__()
		vgg_pretrained_features = torchvision.models.vgg16_bn(pretrained=pretrained).features
		
		self.norm_layer_1 = SPADE(norm_nc=64, label_nc=label_nc)
		self.norm_layer_2 = SPADE(norm_nc=64, label_nc=label_nc)
		self.norm_layer_3 = SPADE(norm_nc=128, label_nc=label_nc)
		self.norm_layer_4 = SPADE(norm_nc=128, label_nc=label_nc)
		self.norm_layer_5 = SPADE(norm_nc=256, label_nc=label_nc)
		self.norm_layer_6 = SPADE(norm_nc=256, label_nc=label_nc)
		self.norm_layer_7 = SPADE(norm_nc=256, label_nc=label_nc)
		self.norm_layer_8 = SPADE(norm_nc=512, label_nc=label_nc)
		self.norm_layer_9 = SPADE(norm_nc=512, label_nc=label_nc)
		self.norm_layer_10 = SPADE(norm_nc=512, label_nc=label_nc)
		
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
		
		for x in range(1):
			self.slice1.add_module(str(x), vgg_pretrained_features[x])
		for x in range(2, 4):
			self.slice2.add_module(str(x), vgg_pretrained_features[x])
		for x in range(5, 6):
			self.slice3.add_module(str(x), vgg_pretrained_features[x])
		for x in range(6, 8):
			self.slice4.add_module(str(x), vgg_pretrained_features[x])
		for x in range(9, 11):
			self.slice5.add_module(str(x), vgg_pretrained_features[x])
		for x in range(12, 13):
			self.slice6.add_module(str(x), vgg_pretrained_features[x])
		for x in range(13, 15):
			self.slice7.add_module(str(x), vgg_pretrained_features[x])
		for x in range(16, 18):
			self.slice8.add_module(str(x), vgg_pretrained_features[x])
		for x in range(19, 21):
			self.slice9.add_module(str(x), vgg_pretrained_features[x])
		for x in range(22, 23):
			self.slice10.add_module(str(x), vgg_pretrained_features[x])
		for x in range(23, 25):
			self.slice11.add_module(str(x), vgg_pretrained_features[x])
		for x in range(26, 28):
			self.slice12.add_module(str(x), vgg_pretrained_features[x])
		for x in range(29, 31):
			self.slice13.add_module(str(x), vgg_pretrained_features[x])
		for x in range(32, 33):
			self.slice14.add_module(str(x), vgg_pretrained_features[x])
	
	def forward(self, img, semantic_img):
		h_relu1 = self.slice3(self.norm_layer_2(self.slice2(self.norm_layer_1(self.slice1(img), semantic_img)), semantic_img))
		h_relu2 = self.slice6(self.norm_layer_4(self.slice5(self.norm_layer_3(self.slice4(h_relu1), semantic_img)), semantic_img))
		h_relu3 = self.slice10(self.norm_layer_7(self.slice9(self.norm_layer_6(self.slice8(self.norm_layer_5(self.slice7(h_relu2), semantic_img)), semantic_img)), semantic_img))
		h_relu4 = self.slice14(self.norm_layer_10(self.slice13(self.norm_layer_9(self.slice12(self.norm_layer_8(self.slice11(h_relu3), semantic_img)), semantic_img)), semantic_img))
		
		out = [h_relu1, h_relu2, h_relu3, h_relu4]
		
		return out

class VGG19_difference(torch.nn.Module):
	def __init__(self, requires_grad=False):
		super().__init__()
		vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
		self.up5 = Upsample(scale_factor=16, mode='bicubic')
		self.up4 = Upsample(scale_factor=8, mode='bicubic')
		self.up3 = Upsample(scale_factor=4, mode='bicubic')
		self.up2 = Upsample(scale_factor=2, mode='bicubic')
		self.up1 = Upsample(scale_factor=1, mode='bicubic')
		self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]


		
		self.slice1 = torch.nn.Sequential()
		self.slice2 = torch.nn.Sequential()
		self.slice3 = torch.nn.Sequential()
		self.slice4 = torch.nn.Sequential()
		self.slice5 = torch.nn.Sequential()
		for x in range(2):
			self.slice1.add_module(str(x), vgg_pretrained_features[x])
		for x in range(2, 7):
			self.slice2.add_module(str(x), vgg_pretrained_features[x])
		for x in range(7, 12):
			self.slice3.add_module(str(x), vgg_pretrained_features[x])
		for x in range(12, 21):
			self.slice4.add_module(str(x), vgg_pretrained_features[x])
		for x in range(21, 30):
			self.slice5.add_module(str(x), vgg_pretrained_features[x])
		if not requires_grad:
			for param in self.parameters():
				param.requires_grad = False
	
	def forward(self, X, Y):
		x1 = self.slice1(X)
		x2 = self.slice2(x1)
		x3 = self.slice3(x2)
		x4 = self.slice4(x3)
		x5 = self.slice5(x4)
		
		y1 = self.slice1(Y)
		y2 = self.slice2(y1)
		y3 = self.slice3(y2)
		y4 = self.slice4(y3)
		y5 = self.slice5(y4)
		
		feat1 = torch.mean(torch.abs(x1-y1), dim=1).unsqueeze(0)
		feat2 = torch.mean(torch.abs(x2-y2), dim=1).unsqueeze(0)
		feat3 = torch.mean(torch.abs(x3-y3), dim=1).unsqueeze(0)
		feat4 = torch.mean(torch.abs(x4-y4), dim=1).unsqueeze(0)
		feat5 = torch.mean(torch.abs(x5-y5), dim=1).unsqueeze(0)
		
		img_5 = self.up5(feat5)
		img_4 = self.up4(feat4)
		img_3 = self.up3(feat3)
		img_2 = self.up2(feat2)
		img_1 = self.up1(feat1)
		perceptual_diff = self.weights[0] * img_1 + self.weights[1] * img_2 + self.weights[2] * img_3 + self.weights[3] * img_4 + self.weights[
    4] * img_5
		
		return perceptual_diff

if __name__ == "__main__":
	from PIL import Image
	import torchvision.models as models
	import torchvision.transforms as transforms
	
	img = Image.open('../../sample_images/zm0002_100000.png')
	vgg_features = VGGFeatures(architecture='vgg16', pretrained=True)
	img_transform = transforms.Compose([transforms.ToTensor()])
	img_tensor = img_transform(img)
	outputs = vgg_features(img_tensor.unsqueeze(0))
	print(img_tensor[0].data.shape)
	print(outputs[0].data.shape)
	print(outputs[1].data.shape)
	print(outputs[2].data.shape)
	print(outputs[3].data.shape)

