import torch.nn as nn

class VGGFeatures(nn.Module):

	def __init__(self, original_model):
		super(VGGFeatures, self).__init__()
		self.layer1 = nn.Sequential(*list(original_model.children())[0][:4])
		self.layer2 = nn.Sequential(*list(original_model.children())[0][4:9])
		self.layer3 = nn.Sequential(*list(original_model.children())[0][9:16])
		self.layer4 = nn.Sequential(*list(original_model.children())[0][16:23])

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

if __name__ == "__main__":
	from PIL import Image
	import torchvision.models as models
	import torchvision.transforms as transforms
	
	img = Image.open('../../sample_images/zm0002_100000.png')
	vgg16 = models.vgg16()
	vgg_features = VGGFeatures(original_model=vgg16)
	img_transform = transforms.Compose([transforms.ToTensor()])
	img_tensor = img_transform(img)
	outputs = vgg_features(img_tensor.unsqueeze(0))
	print(img_tensor[0].data.shape)
	print(outputs[0].data.shape)
	print(outputs[1].data.shape)
	print(outputs[2].data.shape)
	print(outputs[3].data.shape)

