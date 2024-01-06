import os

from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, root_dir, label):
        self.root_dir = root_dir
        self.label = label
        self.path = os.path.join(root_dir, label)
        self.img_name_list = os.listdir(self.path)
        pass

    def __getitem__(self, idx):
        img_name = self.img_name_list[idx]
        img_path = os.path.join(self.path, img_name)

        img = Image.open(img_path)
        label = self.label
        return img, label

    def __len__(self):
        return len(self.img_name_list)


if __name__ == "__main__":
    root_dir = "hymenoptera_data/train"
    ants_label = "ants"
    bees_label = "bees"

    ants_dataset = MyDataset(root_dir, ants_label)
    bees_dataset = MyDataset(root_dir, bees_label)
    train_dataset = ants_dataset + bees_dataset
    print(train_dataset[0])
