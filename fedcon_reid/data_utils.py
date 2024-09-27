from torch.utils.data import Dataset
from PIL import Image
from torchvision import datasets, transforms
import os
import json
import torch
from random_erasing import RandomErasing

class ImageDataset(Dataset):
    def __init__(self, imgs,  transform = None):
        self.imgs = imgs
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        data,label = self.imgs[index]
        return self.transform(Image.open(data)), label


class Data():
    def __init__(self, batch_size, erasing_p, color_jitter, train_all):
        #self.datasets = datasets.split(',')
        self.batch_size = batch_size
        self.erasing_p = erasing_p
        self.color_jitter = color_jitter
        self.train_all = '_all' if train_all else ''
        
    def transform(self):
        transform_train = [
                transforms.Resize((256,128), interpolation=3),
                transforms.Pad(10),
                transforms.RandomCrop((256,128)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]

        transform_val = [
                transforms.Resize(size=(256,128),interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]

        if self.erasing_p > 0:
            transform_train = transform_train + [RandomErasing(probability=self.erasing_p, mean=[0.0, 0.0, 0.0])]

        if self.color_jitter:
            transform_train = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train

        self.data_transforms = {
            'train': transforms.Compose(transform_train),
            'val': transforms.Compose(transform_val),
        }        

    def preprocess_kd_data(self, dataset):
        loader, image_dataset = self.preprocess_one_train_dataset(dataset)
        self.kd_loader = loader


    def preprocess_one_train_dataset(self):
        """preprocess a training dataset, construct a data loader.
        """
        data_path = '../person1'
        data_path = os.path.join(data_path, 'train' + self.train_all)
        image_dataset = datasets.ImageFolder(data_path)
        dataset_sizes = len(image_dataset)

        loader = torch.utils.data.DataLoader(
            ImageDataset(image_dataset.imgs, self.data_transforms['train']), 
            batch_size=self.batch_size,
            shuffle=True, 
            num_workers=0, 
            pin_memory=False)

        return loader, dataset_sizes

    def preprocess_train(self):
        """preprocess training data, constructing train loaders
        """
        #selif.transform()     
        self.trainloader, self.dataset_sizes = self.preprocess_one_train_dataset()

        #return trainloader
        
        
    def preprocess_test(self):
        """preprocess testing data, constructing test loaders
        """
        #self.transform()
        test_dir = '../person1'

        #dataset = test_dir.split('/')[1]
        gallery_dataset = datasets.ImageFolder(os.path.join(test_dir, 'gallery'))
        query_dataset = datasets.ImageFolder(os.path.join(test_dir, 'query'))
    
        gallery_dataset = ImageDataset(gallery_dataset.imgs, self.data_transforms['val'])
        query_dataset = ImageDataset(query_dataset.imgs, self.data_transforms['val'])

        self.testloader = {key: torch.utils.data.DataLoader(
                                                dataset, 
                                                batch_size=self.batch_size,
                                                shuffle=False, 
                                                num_workers=0, 
                                                pin_memory=True) for key, dataset in {'gallery': gallery_dataset, 'query': query_dataset}.items()}

        gallery_cameras, gallery_labels = get_camera_ids(gallery_dataset.imgs)
        self.gallery_meta = {
                'sizes':  len(gallery_dataset),
                'cameras': gallery_cameras,
                'labels': gallery_labels
            }
        print(self.gallery_meta)
        query_cameras, query_labels = get_camera_ids(query_dataset.imgs)
        self.query_meta = {
                'sizes':  len(query_dataset),
                'cameras': query_cameras,
                'labels': query_labels
            }
        print(self.query_meta)
        print('Query Sizes:', self.query_meta['sizes'])
        print('Gallery Sizes:', self.gallery_meta['sizes'])
        
        #return testloader, gallery_meta, query_meta
    
    def preprocess(self):
        self.transform()
        self.preprocess_train()
        self.preprocess_test()

    
def get_camera_ids(img_paths):
    """get camera id and labels by image path
    """
    camera_ids = []
    labels = []
    for path, v in img_paths:# path는 jpg file 객체, v는 그 위의 id 폴더명
        filename = os.path.basename(path)
        info=filename.split('_')
        label=info[0]
        camera_id=info[1][1:]
        labels.append(int(label))
        # labels.append(label)
        camera_ids.append(int(camera_id))
    return camera_ids, labels
