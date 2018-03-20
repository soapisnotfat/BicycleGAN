import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os
from PIL import Image


class Edges2Shoes(Dataset):
    def __init__(self, root, transform, mode='train'):
        self.root = root
        self.transform = transform
        self.mode = mode

        data_dir = os.path.join(root, mode)
        self.file_list = os.listdir(data_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.mode, self.file_list[idx])
        img = Image.open(img_path)
        width, height = img.size[0], img.size[1]

        data = img.crop((0, 0, int(width / 2), height))
        ground_truth = img.crop((int(width / 2), 0, width, height))

        data = self.transform(data)
        ground_truth = self.transform(ground_truth)

        return data, ground_truth


def data_loader(root, batch_size=1, shuffle=True, img_size=128, mode='train'):
    transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    dataset = Edges2Shoes(root, transform, mode=mode)

    if batch_size == 'all':
        batch_size = len(dataset)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             num_workers=4,
                                             drop_last=True)
    data_len = len(dataset)

    return dataloader, data_len
