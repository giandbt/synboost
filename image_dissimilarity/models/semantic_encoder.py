from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
import torch

class SemanticEncoder(nn.Module):
	''' Semantic Encoder as described in Detecting the Unexpected via Image Resynthesis '''
	def __init__(self, in_channels=3):
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
				m.weight.data.normal_(0, 0.01)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
	
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