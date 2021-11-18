
import numpy as np

from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
from torchvision import transforms

from continuum import ClassIncremental
from continuum.datasets import MNIST, CIFAR10, CIFAR100, ImageNet100, ImageNet1000

PATH_DATASETS = "/data/Public/Datasets"

class incremental_scenario(object):
    def __init__(self,
                dataset_name:str,
                train_additional_transforms:list,
                test_additional_transforms:list,
                initial_increment:int,
                increment:int,
                datasets_dir:str = PATH_DATASETS): # 97 server

        super().__init__()

        self.datasets_dir = datasets_dir
        self.dataset_name = dataset_name
        self.train_additional_transforms = train_additional_transforms
        self.test_additional_transforms = test_additional_transforms
        self.initial_increment = initial_increment
        self.increment = increment

        # download datasets and prepare transforms.
        self.prepare_data()
        self.setup()
        
    def prepare_data(self):
        if self.dataset_name == "MNIST":
            self.train_dataset = MNIST(self.datasets_dir, download=True, train=True)
            self.test_dataset = MNIST(self.datasets_dir, download=True, train=False)

        if self.dataset_name == "CIFAR10":
            self.train_dataset = CIFAR10(self.datasets_dir, download=True, train=True)
            self.test_dataset = CIFAR10(self.datasets_dir, download=True, train=False)

        if self.dataset_name == "CIFAR100":
            self.train_dataset = CIFAR100(self.datasets_dir, download=True, train=True)
            self.test_dataset = CIFAR100(self.datasets_dir, download=True, train=False)

        if self.dataset_name == "ImageNet100":
            self.train_dataset = ImageNet100(self.datasets_dir, download=True, train=True)
            self.test_dataset = ImageNet100(self.datasets_dir, download=True, train=False)

        if self.dataset_name == "ImageNet1000": # i.e. ILSVRC2012
            self.train_dataset = ImageNet1000(self.datasets_dir, download=True, train=True)
            self.test_dataset = ImageNet1000(self.datasets_dir, download=True, train=False)
        
   
    def setup(self): # called in every process
        if self.dataset_name == "MNIST":
            self.train_dataset = MNIST(self.datasets_dir, download=False, train=True) # default: train=True
            self.test_dataset = MNIST(self.datasets_dir, download=False, train=False)
            self.train_default_transforms = []
            self.test_default_transforms = []
            self.common_default_transforms = [
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ]
            self.nb_total_classes = 10
            self.dims = (1,28,28)

        if self.dataset_name == "CIFAR10":
            self.train_dataset = CIFAR10(self.datasets_dir, download=False, train=True)
            # print('$'*100, len(self.train_dataset.dataset))
            self.test_dataset = CIFAR10(self.datasets_dir, download=False, train=False)
            self.train_default_transforms = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=63/255)
            ]
            self.test_default_transforms = []
            self.common_default_transforms = [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
            ]
            self.nb_total_classes = 10
            self.dims = (3,32,32)

        if self.dataset_name == "CIFAR100":
            self.train_dataset = CIFAR100(self.datasets_dir, download=False, train=True)
            self.test_dataset = CIFAR100(self.datasets_dir, download=False, train=False)
            self.train_default_transforms = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=63/255)
            ]
            self.test_default_transforms = []
            self.common_default_transforms = [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
            ]
            self.nb_total_classes = 100
            self.dims = (3,32,32)

        if self.dataset_name == "ImageNet100": # TODO @complete all default transforms.
            self.train_dataset = ImageNet100(self.datasets_dir, download=False, train=True)
            self.test_dataset = ImageNet100(self.datasets_dir, download=False, train=False)
            # TODO @temporarily use ImageNet1k's transforms
            self.train_default_transforms = [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=63/255)
            ]
            self.test_default_transforms = [
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ]
            self.common_default_transforms = [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
            self.nb_total_classes = 100
            self.dims = (3,224,224)

        if self.dataset_name == "ImageNet1000": # i.e. ILSVRC2012
            self.train_dataset = ImageNet1000(self.datasets_dir, download=False, train=True)
            self.test_dataset = ImageNet1000(self.datasets_dir, download=False, train=False)
            self.train_default_transforms = [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=63/255)
            ]
            self.test_default_transforms = [
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ]
            self.common_default_transforms = [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
            self.nb_total_classes = 1000
            self.dims = (3,224,224)

    def get_incremental_scenarios(self, logger):
        #order = np.arange(10).tolist()
        self.class_order = np.random.permutation(self.nb_total_classes).tolist()
        logger.info(f'class order: {self.class_order}') # pl seed is the same, here order will be same too.

        train_transforms =  self.train_default_transforms + \
                            self.common_default_transforms + \
                            self.train_additional_transforms

        test_transforms =   self.test_default_transforms + \
                            self.common_default_transforms + \
                            self.test_additional_transforms

        train_scenario = ClassIncremental(
            self.train_dataset,
            initial_increment = self.initial_increment,
            increment = self.increment,
            transformations = train_transforms,
            class_order = self.class_order
        )

        test_scenario = ClassIncremental(
            self.test_dataset,
            initial_increment = self.initial_increment,
            increment = self.increment,
            transformations = test_transforms,
            class_order = self.class_order
        )

        return train_scenario, test_scenario