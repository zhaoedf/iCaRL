
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from pytorch_lightning import LightningDataModule

from continuum import ClassIncremental
from continuum.tasks import split_train_val


from continuum.tasks.task_set import TaskSet

BATCH_SIZE = 64
PATH_DATASETS = "/data/Public/Datasets"

class IncrementalDataModule(LightningDataModule):

    def __init__(self, 
                 task_id:int, 
                 train_taskset:TaskSet, 
                 test_taskset:TaskSet,
                 dims:tuple, 
                 nb_total_classes:int,
                 batch_size:int,
                 num_workers:int,
                 val_split_ratio:float):
        super().__init__()
        self.task_id = task_id
        self.train_taskset = train_taskset
        self.test_taskset = test_taskset

        self.dims = dims
        self.nb_total_classes = nb_total_classes

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split_ratio = val_split_ratio

    def prepare_data(self): # 这步其实就是为了download，即检查数据集是否存在而已。
        pass


    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            full = self.train_taskset
            self.train, self.val = split_train_val(full, val_split=self.val_split_ratio) # train:total*(1-val_split_ratio)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test = self.test_taskset


    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        if self.val_split_ratio == 0.0:
            # return None # return None will force trainer to disable Validation loop.
            self.val = self.test_taskset

        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)