import os
import sys
import numpy as np

import logging
import tqdm

from functools import reduce

import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
from torchmetrics import MeanMetric

from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import MLFlowLogger

from mlflow.tracking import MlflowClient
# import mlflow

# from continuum import ClassIncremental
# from continuum.tasks import split_train_val
# from continuum.datasets import MNIST, CIFAR10
from data.incremental_scenario import incremental_scenario
from continuum import TaskSet

from args.args_trainer import args_trainer
from args.args_model import args_model


from utils.auxiliary_func import setup_logger, get_class_mean



# seed
# seed_everything(args_model.seed, workers=True)

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
# changhong reproduce[github png]
ch_data = [82.9, 75.1, 64.5, 57.1, 52.5]
paper_data = [82.9, 72.3, 67.3, 60.8, 54.4]
for i in range(len(ch_data)):
    mlflow_logger.log_metrics({'ch_data': ch_data[i]}, step=i)
    mlflow_logger.log_metrics({'paper_data': paper_data[i]}, step=i)
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
    datasets_dir = args_model.datasets_dir,
    total_memory_size = args_model.total_memory_size
)
train_scenario, test_scenario, memory = inc_scenario.get_incremental_scenarios(logger)
# memory.herding_method = herd_closest_to_barycenter


# loss function
loss_func = F.binary_cross_entropy_with_logits

# acc & loss metrics
train_epoch_acc = Accuracy().to(device)
test_epoch_acc = Accuracy().to(device)
test_NME_acc = Accuracy().to(device)
train_epoch_loss = MeanMetric().to(device)
train_total_loss = MeanMetric().to(device)


try:
    nb_seen_classes = args_model.initial_increment
    avg_incremental_acc = np.array([])
    old_model = None
    for task_id, taskset in enumerate(train_scenario):
        
        # update clssifier output dim
        print(f'update fc dims to {nb_seen_classes}')
        model.update_fc(nb_seen_classes)
        model = model.to(device)
        
        # add exemplars to train_set
        if task_id > 0:
            mem_x, mem_y, mem_t = memory.get()
            taskset.add_samples(mem_x, mem_y, mem_t)
        
        # data
        train_set = taskset
        test_set = test_scenario[:task_id+1]
        
        train_loader = torch.utils.data.DataLoader(
            dataset=taskset,
            batch_size=args_model.batch_size,
            shuffle=True,
            num_workers=args_model.num_workers,
            drop_last=False
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=args_model.batch_size,
            shuffle=True,
            num_workers=args_model.num_workers,
            drop_last=False
        )
        nme_test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=args_model.batch_size,
            shuffle=True,
            num_workers=args_model.num_workers,
            drop_last=False
        )

        # optimizer & scheduler
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args_model.learning_rate, momentum=0.9, weight_decay=5e-5
        )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[49,63], gamma=0.2
        )
        
        for epoch in range(args_trainer.max_epochs):
            
            # train
            model.train()
            for idx, batch in enumerate(train_loader):
                x, y, t = batch
                x, y = x.to(device), y.to(device) # changhong style
                logits = model(x)['logits']
                y_one_hot = F.one_hot(y, num_classes=logits.shape[1]).type_as(logits)  # expand_as 更优雅 TODO
                
                if old_model is None:
                    loss = loss_func(logits, y_one_hot)
                else:
                    old_onehots = torch.sigmoid(old_model(x)['logits'].detach())
                    new_onehots = y_one_hot.clone()
                    # logger.info(f'{new_onehots.shape} {old_onehots.shape}')
                    new_onehots[:, :nb_seen_classes-args_model.increment] = old_onehots
                    loss = loss_func(logits, new_onehots)
                
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
            logger.info(f'{lr_scheduler.get_last_lr()}')
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
            # model.train()
            
            # logger.info(f'Task: {task_id}, Epoch: {epoch+1}/{args_trainer.max_epochs}')
            # logger.info(f'avg_test_acc for [0:{nb_seen_classes}]  {avg_incremental_acc[-1]}')
            acc_epoch = test_epoch_acc.compute().item()
            logger.info(f'test_epoch_acc => {acc_epoch}')
            mlflow_logger.log_metrics({f'task{task_id}_test_epoch_acc': acc_epoch}, step=epoch)
            if epoch+1 == args_trainer.max_epochs:
                logger.info('@'*50)
                logger.info(f'one increment end')
                avg_incremental_acc = np.append(avg_incremental_acc, acc_epoch)
                logger.info(f'avg_incremental_acc for [0:{nb_seen_classes}] => {avg_incremental_acc[-1]}')  # avg_incremental_acc.mean()
                mlflow_logger.log_metrics({'avg_incremental_acc': avg_incremental_acc[-1]*100}, step=task_id)
                logger.info('@'*50)
            
            # client.log_metric(run_id, key='avg_test_acc', value=test_Accuracy.compute().item(), step=task_id)
            # client.log_metric(run_id, key='avg_incremental_acc', value=avg_incremental_acc.mean(), step=task_id)
            train_epoch_loss.reset()
            train_epoch_acc.reset()
            test_epoch_acc.reset()
            logger.info('*'*75)
        
        # constructing new exemplar set
        # model.eval()
        # get_extract_features = lambda x, model=model: model.extract_vector(x)
        features = []
        temp_set = TaskSet(*train_scenario[task_id].get_raw_samples(), test_set.trsf, data_type=test_set.data_type)
        loader =  torch.utils.data.DataLoader(temp_set, shuffle=False, batch_size=args_model.batch_size)
        for idx, (x,y,t) in enumerate(loader):
            # print(y.shape)
            features.append(model.extract_vector(x.to(device)).cpu().detach().numpy())
        features = np.concatenate(tuple(features))
        # logger.info(f'{len(features)}')
        memory.add(
            *train_scenario[task_id].get_raw_samples(), features
        )
        old_model = model.copy().freeze()
        # logger.info(f'{memory.memory_per_class}')
        
        # NME prediction
        all_features = []
        # if task_id > 0:
        mem_x, mem_y, mem_t = memory.get()
        memory_set = TaskSet(mem_x, mem_y, mem_t, test_set.trsf, data_type=test_set.data_type)
        # print(taskset.data_type) # TaskType.IMAGE_ARRAY

        another_loader = torch.utils.data.DataLoader(memory_set, shuffle=False, batch_size=args_model.batch_size)
        for idx, (x,y,t) in enumerate(another_loader):
            # print(x.shape)
            all_features.append(model.extract_vector(x.to(device)).cpu().detach().numpy())
        all_features = np.concatenate(tuple(all_features))
        class_means = np.expand_dims(get_class_mean(mem_y, all_features), axis=0) # .get() returns x,y,z
        # else:
        #     class_means = np.expand_dims(get_class_mean(train_scenario[task_id].get_raw_samples()[1], features), axis=0)
        
        correct = 0
        total = 0
        for idx, batch in enumerate(nme_test_loader):
            x, y, t = batch
            # x, y = x.to(device), y.to(device)
            feature_vector = np.expand_dims(model.extract_vector(x.to(device)).cpu().detach().numpy(), axis=1) # (bs, feature_dim)
            dist_to_mean = np.linalg.norm(class_means - feature_vector, axis=2) # (bs, nb_classes)
            preds = dist_to_mean.argsort()[:, 0]  # default: ascend    
            # print(dist_to_mean.shape, preds.shape,(class_means - feature_vector).shape)    
            correct += np.count_nonzero(preds==y.numpy())
            total += y.shape[0]
            # print(correct, preds[0], y[0])
        nme_acc = (correct / total)*100
        logger.info(f'{nme_acc}, {correct}')
        mlflow_logger.log_metrics({'nme_acc': nme_acc}, step=task_id)
            # x, y = x.to(device), y.to(device) # changhong style
            # y_hat = torch.sigmoid(model(x)['logits']) 
            # y_one_hot = F.one_hot(y, num_classes=y_hat.shape[1]).type_as(y_hat)
            # loss = F.binary_cross_entropy(y_hat, y_one_hot)  # binary_cross_entropy_with_logits is loss func with sigmoid inside.
            # test_epoch_acc.update(y_hat, y)

        nb_seen_classes += args_model.increment
    
    mlflow_logger.experiment.log_artifact(run_id, os.path.join(log_path, run_id)+'.log')
    mlflow_logger.finalize(status='FINISHED')
except KeyboardInterrupt:
    print('hello')
    mlflow_logger.finalize(status='KILLED')
    # mlflow_logger.experiment.set_terminated(run_id, "KILLED")
    mlflow_logger.experiment.log_artifact(run_id, os.path.join(log_path, run_id)+'.log')


# client.set_terminated(run_id=run_id, status='FINISHED')