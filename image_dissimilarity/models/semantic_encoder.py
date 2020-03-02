import torch.nn as nn

class SemanticEncoder(nn.Module):
	''' Semantic Encoder as described in Detecting the Unexpected via Image Resynthesis '''
	
	def __init__(self, in_channels=19):
		super(SemanticEncoder, self).__init__()
		self.relu = nn.ReLU(True)
		self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=7, stride=1, padding=3)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
		self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
		self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
		
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
		x = self.conv1(x)
		x1 = self.relu(x)
		x = self.conv2(x1)
		x2 = self.relu(x)
		x = self.conv3(x2)
		x3 = self.relu(x)
		x = self.conv4(x3)
		x4 = self.relu(x)
		return [x1, x2, x3, x4]


if __name__ == "__main__":
	from PIL import Image
	import torchvision.transforms as transforms
	
	img = Image.open('../../sample_images/zm0002_100000.png')
	test = SemanticEncoder()
	img_transform = transforms.Compose([transforms.ToTensor()])
	img_tensor = img_transform(img)
	outputs = test(img_tensor.unsqueeze(0))
	print(img_tensor[0].data.shape)
	print(outputs[0].data.shape)
	print(outputs[1].data.shape)
	print(outputs[2].data.shape)
	print(outputs[3].data.shape)
