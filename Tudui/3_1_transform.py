from torchvision import transforms
from PIL import Image

img = Image.open("hymenoptera_data/train/ants/6240338_93729615ec.jpg")

transform = transforms.ToTensor()
img_tensor = transform(img)