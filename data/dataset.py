import os.path
import random
import torchvision.transforms as transforms
import torch
import torch.utils.data as data
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(directory):
    images = []
    assert os.path.isdir(directory), '%s is not a valid directory' % directory

    for root, _, filenames in sorted(os.walk(directory)):
        for filename in filenames:
            if is_image_file(filename):
                path = os.path.join(root, filename)
                images.append(path)

    return images


class SingleDataset(data.Dataset):
    def __init__(self, opt):
        super(SingleDataset, self).__init__()
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot)

        self.A_paths = make_dataset(self.dir_A)

        self.A_paths = sorted(self.A_paths)

        self.transform = get_transform(opt)

    def __getitem__(self, index):
        a_path = self.A_paths[index]
        a_img = Image.open(a_path).convert('RGB')
        a_ = self.transform(a_img)
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
        else:
            input_nc = self.opt.input_nc

        if input_nc == 1:  # RGB to gray
            tmp = a_[0, ...] * 0.299 + a_[1, ...] * 0.587 + a_[2, ...] * 0.114
            a_ = tmp.unsqueeze(0)

        return {'A': a_, 'A_paths': a_path}

    def __len__(self):
        return len(self.A_paths)


class AlignedDataset(data.Dataset):
    def __init__(self, opt):
        super(AlignedDataset, self).__init__()
        self.opt = opt
        self.root = opt.dataroot
        self.center_crop = opt.center_crop
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.ab_paths = sorted(make_dataset(self.dir_AB))
        assert (opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        ab_path = self.ab_paths[index]
        a_b = Image.open(ab_path).convert('RGB')
        a_b = a_b.resize(
            (self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
        a_b = transforms.ToTensor()(a_b)

        w_total = a_b.size(2)
        w = int(w_total / 2)
        h = a_b.size(1)
        if self.center_crop:
            w_offset = int(round((w - self.opt.fineSize) / 2.0))
            h_offset = int(round((h - self.opt.fineSize) / 2.0))
        else:
            w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
            h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        a_ = a_b[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        b_ = a_b[:, h_offset:h_offset + self.opt.fineSize, w + w_offset:w + w_offset + self.opt.fineSize]
        a_ = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a_)
        b_ = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b_)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if not self.opt.no_flip and random.random() < 0.5:
            idx = [i for i in range(a_.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            a_ = a_.index_select(2, idx)
            b_ = b_.index_select(2, idx)

        if input_nc == 1:
            tmp = a_[0, ...] * 0.299 + a_[1, ...] * 0.587 + a_[2, ...] * 0.114
            a_ = tmp.unsqueeze(0)

        if output_nc == 1:
            tmp = b_[0, ...] * 0.299 + b_[1, ...] * 0.587 + b_[2, ...] * 0.114
            b_ = tmp.unsqueeze(0)

        return {'A': a_, 'B': b_,
                'A_paths': ab_path, 'B_paths': ab_path}

    def __len__(self):
        return len(self.ab_paths)


def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __scale_width(img, target_width):
    ow, oh = img.size
    if ow == target_width:
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)
