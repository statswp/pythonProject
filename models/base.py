import datetime
import time
import warnings

import numpy as np
import torch
import torchmetrics
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from utils.inputs import DenseFeature, SparseFeature, VarlenFeature
from utils.inputs import create_feature_index, create_embed_dict
from utils.inputs import dense_feature_to_input, sparse_feature_to_input, varlen_feature_to_inputs

warnings.filterwarnings("ignore")


class BaseModel(nn.Module):
    def __init__(self,
                 feature_columns,
                 name='BaseModel'
                 ):
        super(BaseModel, self).__init__()
        self.feature_columns = feature_columns
        self.name = name

        self.embed_dict = create_embed_dict(feature_columns)
        self.feature_index = create_feature_index(feature_columns)
        self.dense_feature_columns = [f for f in feature_columns if isinstance(f, DenseFeature)]
        self.sparse_feature_columns = [f for f in feature_columns if
                                       isinstance(f, SparseFeature) and not isinstance(f, VarlenFeature)]
        self.varlen_feature_columns = [f for f in feature_columns if isinstance(f, VarlenFeature)]

        self.optimizer = None
        self.loss_fn = None
        self.summary = None
        self.metrics = None

    def compute_in_features(self):
        in_features = 0
        for f in self.dense_feature_columns:
            in_features += f.dim
        for f in self.sparse_feature_columns:
            in_features += f.embed_dim
        for f in self.varlen_feature_columns:
            in_features += f.embed_dim
        return in_features

    def forward(self, x):
        dense_inputs = dense_feature_to_input(x, self.feature_index,
                                              self.dense_feature_columns)
        sparse_inputs = sparse_feature_to_input(x, self.embed_dict, self.feature_index,
                                                self.sparse_feature_columns)
        varlen_inputs = varlen_feature_to_inputs(x, self.embed_dict, self.feature_index,
                                                 self.varlen_feature_columns)
        pass

    def compile(self, optimizer: torch.optim.Optimizer, loss_fn, metrics=None):
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics

    def fit(self, dataloader, validate_dataloader=None, epochs=100, log_dir=None, log_interval=100):
        loss_fn = self.loss_fn
        optim = self.optimizer
        if log_dir:
            version = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            self.summary = SummaryWriter(log_dir + '/' + self.name + '_' + version)

        global_step = 0
        for epoch in range(epochs):
            print(f"Start Epoch: {epoch}")
            start_time = time.time()
            model = self.train()

            epoch_loss = []
            for step, (x, y_true) in enumerate(dataloader):
                x = x.float()
                y_true = y_true.float()

                y_pred = model(x)

                optim.zero_grad()
                batch_loss = loss_fn(y_pred.squeeze(), y_true)

                batch_loss.backward()
                optim.step()

                epoch_loss.append(batch_loss.item())

                if log_dir and global_step % log_interval == 0:
                    self.summary.add_scalar("Loss/train", batch_loss.item(), global_step)
                    print(f"Epoch: {epoch}, global_step: {global_step}, loss: {batch_loss.item()}")

                global_step += 1

            # 验证
            if validate_dataloader:
                eval_metric = self.evaluate(validate_dataloader)
                print(f"Epoch: {epoch}, global_step: {global_step}, eval_loss: {eval_metric['eval_loss']}, "
                      f"eval_auc: {eval_metric['eval_auc']}")

                if log_dir:
                    self.summary.add_scalar("Loss/test", eval_metric['eval_loss'], global_step)
                    self.summary.add_scalar("AUC/test", eval_metric['eval_auc'], global_step)

            end_time = time.time()
            seconds = int(end_time - start_time)
            print(f"Epoch: {epoch}, seconds: {seconds}")

    def evaluate(self, dataloader):
        model = self.eval()
        auc_metric = torchmetrics.AUROC(task='binary')
        loss = []
        with torch.no_grad():
            for x, y in dataloader:
                x = x.float()
                y = y.float()
                y_ = model(x).squeeze()

                cur_loss = self.loss_fn(y_, y)
                loss.append(cur_loss.item())

                auc_metric(y_, y)

        auc = auc_metric.compute()

        return {
            'eval_loss': np.mean(loss),
            'eval_auc': auc
        }

    def predict(self, dataloader):
        model = self.eval()
        pred_ans = []
        with torch.no_grad():
            for x, y in dataloader:
                y_pred = model(x.float())
                pred_ans.append(y_pred)

        pred_ans = torch.cat(pred_ans, dim=0)
        return pred_ans
