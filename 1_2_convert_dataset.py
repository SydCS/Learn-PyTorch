import os

root_dir = "hymenoptera_data/train"
img_dir = "ants"
img_name_list = os.listdir(os.path.join(root_dir, img_dir))

label = img_dir
label_dir = "ants_label"
for i in img_name_list:
    img_name = i.split(".jpg")[0]
    with open(os.path.join(root_dir, label_dir, f'{img_name}.txt'), "w") as f:
        f.write(label)
