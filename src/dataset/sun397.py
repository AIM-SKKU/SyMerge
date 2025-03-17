import os
import torch
from datasets import load_dataset
import torchvision.datasets as datasets
from datasets import Dataset


# PyTorch 호환 데이터셋 클래스 정의
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
    


class SUN397:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=0):
        # Data loading code
        traindir = os.path.join(location, 'sun397', 'data')
        valdir = os.path.join(location, 'sun397', 'data')

        data_path = os.path.join(location, 'sun397')

        data = load_dataset(data_path)

        self.train_dataset = data['train']
        self.classnames = self.train_dataset.features['label'].names
        # ArrowDataset을 PyTorch 데이터셋으로 변환
        self.train_dataset = ArrowImageDataset(self.train_dataset, transform=preprocess)



        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = data['test']
        # ArrowDataset을 PyTorch 데이터셋으로 변환
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
