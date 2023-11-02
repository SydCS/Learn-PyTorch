from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")

# for i in range(100):
#     writer.add_scalar("y=2x", 2 * i, i)

image = Image.open("hymenoptera_data/train/ants/6240338_93729615ec.jpg")
image_array = np.array(image)
writer.add_image("test", image_array, 1, dataformats="HWC")

writer.close()
