import torch.utils.data as data


class DataLoader(object):
    def __init__(self, opt):
        self.opt = opt
        self.dataset = self.create_dataset()
        self.dataloader = data.DataLoader(self.dataset, batch_size=opt.batchSize, shuffle=not opt.serial_batches, num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def create_dataset(self):
        if self.opt.dataset_mode == 'aligned':
            from data.dataset import AlignedDataset
            dataset = AlignedDataset(self.opt)
        elif self.opt.dataset_mode == 'single':
            from data.dataset import SingleDataset
            dataset = SingleDataset(self.opt)
        else:
            raise ValueError("Dataset [%s] not recognized." % self.opt.dataset_mode)

        return dataset

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)


def create_data_loader(opt):
    data_loader = DataLoader(opt)
    return data_loader
