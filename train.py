import os
import sys
import numpy as np

import logging
import tqdm

import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
from torchmetrics import MeanMetric

from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import MLFlowLogger

from mlflow.tracking import MlflowClient
import mlflow

# from continuum import ClassIncremental
# from continuum.tasks import split_train_val
# from continuum.datasets import MNIST, CIFAR10
from data.incremental_scenario import incremental_scenario

from args.args_trainer import args_trainer
from args.args_model import args_model
# TODO  mlfow save hyperparams


from utils.auxiliary_func import setup_logger



# seed
seed_everything(args_model.seed, workers=True)

# gpu
device = "cuda:"+args_trainer.gpus if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# changhong code
from incremental_net.inc_net import IncrementalNet
model = IncrementalNet(args_model.backbone, pretrained=False, gradcam=False)

# mlflow
exp_name = "incremental_learning"
mlflow_logger = MLFlowLogger(experiment_name=exp_name, tracking_uri="http://localhost:10500")
run_id = mlflow_logger.run_id
# client = MlflowClient(tracking_uri='http://localhost:10500')
# exp_id = client.get_experiment_by_name("incremental_learning").experiment_id
# run_id = client.list_run_infos(exp_id)[0].run_id
mlflow_logger.log_hyperparams(args_model)
# mlflow_logger.log_hyperparams(args_trainer)
# logging
log_path = './logs'
if not os.path.exists(log_path):
    os.mkdir(log_path)

logger = setup_logger(log_path=log_path, mlflow_runid=run_id)

# incremenal dataset via cotinuum
inc_scenario = incremental_scenario(
    dataset_name = args_model.dataset,
    train_additional_transforms = [],
    test_additional_transforms = [],
    initial_increment = args_model.initial_increment,
    increment = args_model.increment,
    datasets_dir = args_model.datasets_dir
)
train_scenario, test_scenario = inc_scenario.get_incremental_scenarios(logger)



# optimizer & scheduler
optimizer = torch.optim.SGD(
    model.parameters(), lr=args_model.learning_rate, momentum=0.9, weight_decay=5e-5
)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[49,63], gamma=0.2
)

# loss function
loss_func = F.binary_cross_entropy_with_logits

# acc & loss metrics
train_epoch_acc = Accuracy().to(device)
test_epoch_acc = Accuracy().to(device)
train_epoch_loss = MeanMetric().to(device)
train_total_loss = MeanMetric().to(device)



nb_seen_classes = args_model.initial_increment
avg_incremental_acc = np.array([])
for task_id, taskset in enumerate(train_scenario):
    
    # update clssifier output dim
    print(f'update fc dims to {nb_seen_classes}')
    model.update_fc(nb_seen_classes)
    model = model.to(device)
    
    # data
    train_set = taskset
    test_set = test_scenario[:task_id+1]
    
    train_loader = torch.utils.data.DataLoader(
        dataset=taskset,
        batch_size=args_model.batch_size,
        shuffle=True,
        num_workers=args_model.num_workers,
        drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args_model.batch_size,
        shuffle=True,
        num_workers=args_model.num_workers,
        drop_last=True
    )


    
    for epoch in range(args_trainer.max_epochs):

        # train
        for idx, batch in enumerate(train_loader):
            x, y, t = batch
            x, y = x.to(device), y.to(device) # changhong style
            logits = model(x)['logits']
            y_one_hot = F.one_hot(y, num_classes=logits.shape[1]).type_as(logits)
            loss = loss_func(logits, y_one_hot)
            
            train_epoch_loss.update(loss)
            train_total_loss.update(loss)
            train_epoch_acc.update(torch.sigmoid(logits), y) # y_hat: [bs, nb_classes]   y: [bs,]  [checked]
            # if task_id == 1: logger.info(f'{y.shape}, {logits.shape}, {y.unique()}')


            # if idx % args_trainer.log_every_n_steps == 0:
            #     logger.info(f'Epoch: {epoch}/{args_trainer.max_epochs} batch: {idx}, loss: {loss.item()}')
            #     logger.info(f'train_acc: {train_acc}')

            # update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # update lr_scheduler
        lr_scheduler.step() # should update every epoch instead of step! otherwise it will decay based on step instead of epoch! see source code in doc.
    
        logger.info('*'*75)
        # logger.info('one epoch end')
        logger.info(f'Task: {task_id}, Epoch: {epoch+1}/{args_trainer.max_epochs}')
        logger.info(f'train_epoch_loss => {train_epoch_loss.compute().item()} train_total_loss => {train_total_loss.compute().item()}')
        logger.info(f'train_epoch_acc => {train_epoch_acc.compute()} ')
        logger.info('*'*75)
        mlflow_logger.log_metrics({f'task{task_id}_train_epoch_loss': train_total_loss.compute().item()}, step=epoch)
        mlflow_logger.log_metrics({f'task{task_id}_train_epoch_acc': train_epoch_acc.compute().item()}, step=epoch)
        # client.log_metric(run_id, key=f'task{task_id}_train_epoch_loss', value=train_total_loss.compute().item(), step=epoch)
        # client.log_metric(run_id, key=f'task{task_id}_train_epoch_acc', value=train_Accuracy.compute().item(), step=epoch)

    
    
        # logger.info('*'*75)
        # logger.info(f'one increment end')
        # test
        model.eval()
        for idx, batch in enumerate(test_loader):
            x, y, t = batch
            x, y = x.to(device), y.to(device) # changhong style
            y_hat = torch.sigmoid(model(x)['logits']) 
            y_one_hot = F.one_hot(y, num_classes=y_hat.shape[1]).type_as(y_hat)
            loss = F.binary_cross_entropy(y_hat, y_one_hot)  # binary_cross_entropy_with_logits is loss func with sigmoid inside.
            test_epoch_acc.update(y_hat, y)
        model.train()
        
        # logger.info(f'Task: {task_id}, Epoch: {epoch+1}/{args_trainer.max_epochs}')
        # logger.info(f'avg_test_acc for [0:{nb_seen_classes}]  {avg_incremental_acc[-1]}')
        acc_epoch = test_epoch_acc.compute().item()
        logger.info(f'test_epoch_acc => {acc_epoch}')
        mlflow_logger.log_metrics({f'task{task_id}_test_epoch_acc': acc_epoch}, step=epoch)
        if epoch+1 == args_trainer.max_epochs:
            logger.info('@'*50)
            logger.info(f'one increment end')
            avg_incremental_acc = np.append(avg_incremental_acc, acc_epoch)
            logger.info(f'avg_incremental_acc for [0:{nb_seen_classes}] => {avg_incremental_acc.mean()}')
            mlflow_logger.log_metrics({'avg_incremental_acc': avg_incremental_acc.mean()}, step=task_id)
            logger.info('@'*50)
        
        # client.log_metric(run_id, key='avg_test_acc', value=test_Accuracy.compute().item(), step=task_id)
        # client.log_metric(run_id, key='avg_incremental_acc', value=avg_incremental_acc.mean(), step=task_id)
        train_epoch_loss.reset()
        train_epoch_acc.reset()
        test_epoch_acc.reset()
        logger.info('*'*75)
    
    
    nb_seen_classes += args_model.increment
    
    

mlflow_logger.finalize(status='FINISHED')
# client.set_terminated(run_id=run_id, status='FINISHED')