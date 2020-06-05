import torch.nn as nn

class SemanticEncoder(nn.Module):
	''' Semantic Encoder as described in Detecting the Unexpected via Image Resynthesis '''
	
	def __init__(self, architecture='vgg16', in_channels=19, num_hidden_layers=4, base_feature_size=32):
		super(SemanticEncoder, self).__init__()
		
		self.hidden_layers = nn.ModuleList()
		
		if 'bn' in architecture:
			for idx in range(num_hidden_layers):
				if idx == 0:
					self.hidden_layers.append(nn.Sequential(
					nn.Conv2d(in_channels, base_feature_size, kernel_size=7, padding=3, stride=1),
					nn.BatchNorm2d(base_feature_size),
					nn.ReLU(inplace=True)))
					in_channels = base_feature_size
				else:
					feature_size = in_channels*2
					self.hidden_layers.append(nn.Sequential(
						nn.Conv2d(in_channels, feature_size, kernel_size=3, padding=1, stride=2),
						nn.BatchNorm2d(feature_size),
						nn.ReLU(inplace=True)))
					in_channels = feature_size
		else:
			for idx in range(num_hidden_layers):
				if idx == 0:
					self.hidden_layers.append(nn.Sequential(
					nn.Conv2d(in_channels, base_feature_size, kernel_size=7, padding=3, stride=1),
					nn.ReLU(inplace=True)))
					in_channels = base_feature_size
				else:
					feature_size = in_channels*2
					self.hidden_layers.append(nn.Sequential(
						nn.Conv2d(in_channels, feature_size, kernel_size=3, padding=1, stride=2),
						nn.ReLU(inplace=True)))
					in_channels = feature_size
				
		self._initialize_weights()
	
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
	
	def forward(self, x):
		output = []
		for idx, layer in enumerate(self.hidden_layers):
			x = layer(x)
			output.append(x)

		return output


class ResNetSemanticEncoder(nn.Module):
	''' Semantic Encoder as described in Detecting the Unexpected via Image Resynthesis '''
	
	def __init__(self, in_channels=19, num_hidden_layers=4):
		super(ResNetSemanticEncoder, self).__init__()
		
		self.hidden_layers = nn.ModuleList()
		base_feature_size = 32
		
		for idx in range(num_hidden_layers):
			if idx == 0:
				self.hidden_layers.append(nn.Sequential(
					nn.Conv2d(in_channels, base_feature_size, kernel_size=7, padding=3, stride=2),
					nn.BatchNorm2d(base_feature_size),
					nn.ReLU(inplace=True),
					nn.MaxPool2d(kernel_size=3, stride=2, padding=1)))
				in_channels = base_feature_size
			else:
				feature_size = in_channels * 2
				self.hidden_layers.append(nn.Sequential(
					nn.Conv2d(in_channels, feature_size, kernel_size=3, padding=1, stride=2),
					nn.BatchNorm2d(feature_size),
					nn.ReLU(inplace=True)))
				in_channels = feature_size

		
		self._initialize_weights()
	
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
	
	def forward(self, x):
		output = []
		for idx, layer in enumerate(self.hidden_layers):
			x = layer(x)
			output.append(x)
		
		return output


if __name__ == "__main__":
	from PIL import Image
	import torchvision.transforms as transforms
	
	img = Image.open('../../sample_images/zm0002_100000.png')
	test = ResNetSemanticEncoder(in_channels=3)
	img_transform = transforms.Compose([transforms.ToTensor()])
	img_tensor = img_transform(img)
	outputs = test(img_tensor.unsqueeze(0))
	print(img_tensor[0].data.shape)
	print(outputs[0].data.shape)
	print(outputs[1].data.shape)
	print(outputs[2].data.shape)
	print(outputs[3].data.shape)
