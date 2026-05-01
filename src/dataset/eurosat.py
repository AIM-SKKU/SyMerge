import os
import torch
import torchvision.datasets as datasets
import re

from datasets import load_dataset
from datasets import Dataset


class ArrowImageDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        # print(item) # {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1024x768 at 0x7F918328F100>, 'label': 0}
        
        image = item['image']  
        
        if self.transform:
            image = self.transform(image)
        
        label = item['label']
        return image, label


def pretify_classname(classname):
    l = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', classname)
    l = [i.lower() for i in l]
    out = ' '.join(l)
    if out.endswith('al'):
        return out + ' area'
    return out

class EuroSATBase:
    def __init__(self,
                 preprocess,
                 test_split,
                 location='~/data',
                 batch_size=32,
                 num_workers=0):
        # Data loading code
        # traindir = os.path.join(location, 'EuroSAT_splits', 'train')
        # testdir = os.path.join(location, 'EuroSAT_splits', test_split)

        data_path = os.path.join(location, 'eurosat')

        data = load_dataset(data_path)

        # self.train_dataset = datasets.ImageFolder(traindir, transform=preprocess)
        self.train_dataset = data['train']
        self.classnames = self.train_dataset.features['label'].names
        self.train_dataset = ArrowImageDataset(self.train_dataset, transform=preprocess)
            
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        # self.test_dataset = datasets.ImageFolder(testdir, transform=preprocess)
        self.test_dataset = data['test']
        self.test_dataset = ArrowImageDataset(self.test_dataset, transform=preprocess)
                 
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        self.test_loader_shuffle = torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers
        )
        # idx_to_class = dict((v, k)
        #                     for k, v in self.train_dataset.class_to_idx.items())

        # self.classnames = [idx_to_class[i].replace('_', ' ') for i in range(len(idx_to_class))]
        # print(self.classnames)
        # # self.classnames = [pretify_classname(c) for c in self.classnames]
        # # print(self.classnames)
        # # ours_to_open_ai = {
        # #     'annual crop': 'annual crop land',
        # #     'forest': 'forest',
        # #     'herbaceous vegetation': 'brushland or shrubland',
        # #     'highway': 'highway or road',
        # #     'industrial area': 'industrial buildings or commercial buildings',
        # #     'pasture': 'pasture land',
        # #     'permanent crop': 'permanent crop land',
        # #     'residential area': 'residential buildings or homes or apartments',
        # #     'river': 'river',
        # #     'sea lake': 'lake or sea',
        # # }
        # # for i in range(len(self.classnames)):
        # #     self.classnames[i] = ours_to_open_ai[self.classnames[i]]


class EuroSAT(EuroSATBase):
    def __init__(self,
                 preprocess,
                 location='~/data',
                 batch_size=32,
                 num_workers=0):
        super().__init__(preprocess, 'test', location, batch_size, num_workers)


class EuroSATVal(EuroSATBase):
    def __init__(self,
                 preprocess,
                 location='~/data',
                 batch_size=32,
                 num_workers=0):
        super().__init__(preprocess, 'val', location, batch_size, num_workers)
