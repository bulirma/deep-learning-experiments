from typing import Iterable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, MeanMetric
from tqdm import tqdm

from traineval import DEVICE


class Model(nn.Module):
    def __init__(self, device, layers: Iterable, use_validation: bool = False):
        super().__init__()
        self.device = device
        self.layers = nn.ModuleList(layers)
        self.use_validation = use_validation
        self.to(self.device)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def configure(self, optimizer, scheduler, loss, metrics: dict):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.test_accuracy = metrics.get('accuracy')
        self.test_loss = metrics.get('loss')
        self.validation_accuracy = metrics.get('validation_accuracy')
        self.validation_loss = metrics.get('validation_loss')
        if self.use_validation and self.validation_accuracy is None:
            raise ValueError('model is set to use validation but no validation accuracy metric was provided')
        if self.use_validation and self.validation_loss is None:
            raise ValueError('model is set to use validation but no validation loss metric was provided')

    def fit(self, num_epochs: int, train_loader: DataLoader, validation_loader: DataLoader = None):
        if self.use_validation and validation_loader is None:
            raise ValueError('model is set to use validation but validation set loader was not provided')

        logs = []
        for e in range(1, num_epochs + 1):

            if self.test_accuracy is not None:
                self.test_accuracy.reset()
            if self.test_loss is not None:
                self.test_loss.reset()
            if self.use_validation:
                self.validation_accuracy.reset()
                self.validation_loss.reset()

            self.train()
            with tqdm(train_loader, unit='batch', desc=f'epoch {e}/{num_epochs}') as pbar:

                for images, targets in pbar:
                    images = images.to(self.device)
                    targets = targets.to(self.device)

                    self.optimizer.zero_grad()
                    predictions = self(images)
                    loss = self.loss(predictions, targets)
                    loss.backward()
                    self.optimizer.step()
                    if not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step()

                    self.test_accuracy.update(predictions, targets)
                    self.test_loss.update(loss)

                    loss = None if self.test_loss is None else '{:.4f}'.format(self.test_loss.compute().item())
                    acc = None if self.test_accuracy is None else '{:.2f}'.format(self.test_accuracy.compute().item())
                    lr = None if self.optimizer is None else '{:.2e}'.format(self.optimizer.param_groups[0]['lr'])

                    pbar.set_postfix(loss=loss, acc=acc, lr=lr)

            log = {
                'acc': self.test_accuracy.compute().item(),
                'loss': self.test_loss.compute().item()
            }

            if self.use_validation:

                self.eval()
                with tqdm(validation_loader, unit='batch', desc=f'epoch {e}/{num_epochs}') as pbar:

                    for images, targets in pbar:
                        images = images.to(self.device)
                        targets = targets.to(self.device)

                        predictions = self(images)
                        loss = self.loss(predictions, targets)

                        self.validation_accuracy.update(predictions, targets)
                        self.validation_loss.update(loss)

                        loss = None if self.validation_loss is None else f'{self.validation_loss.compute().item():.4f}'
                        acc = None if self.validation_accuracy is None else f'{self.validation_accuracy.compute().item():.2f}'

                        pbar.set_postfix(loss=loss, acc=acc)

                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()

                log = {
                    **log,
                    'val_acc': self.validation_accuracy.compute().item(),
                    'val_loss': self.validation_loss.compute().item()
                }

            logs.append(log)

        return logs

    @torch.no_grad()
    def evaluate(self, test_loader):
        self.eval()

        self.test_accuracy.reset()
        self.test_loss.reset()

        for images, targets in test_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            predictions = self(images)
            loss = self.test_loss(predictions, targets)
            
            self.test_accuracy.update(predictions, targets)
            self.test_loss.update(loss)

        return {
            'accuracy': self.test_accuracy.compute().item(),
            'loss': self.test_loss.compute().item()
        }

    @torch.no_grad()
    def predictions(self, image):
        self.eval()
        logits = self(image)
        return torch.nn.functional.softmax(logits, dim=1)

    def predict(self, image):
        probs = self.predictions(image)
        return probs.argmax(dim=1).item()

def setup_byclass_cnn_model(num_classes: int, learning_rate: float, weight_decay: float, use_validation: bool = False):
    model = Model(DEVICE, (
        nn.Conv2d(1, 32, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Conv2d(32, 32, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(2, 2),
        nn.Dropout2d(0.2),
        nn.Conv2d(32, 64, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Conv2d(64, 64, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(2, 2),
        nn.Dropout2d(0.25),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    ))
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if use_validation:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0)
    model.configure(
        optimizer,
        scheduler,
        torch.nn.CrossEntropyLoss(),
        { 
            'accuracy': Accuracy('multiclass', num_classes=num_classes),
            'loss': MeanMetric('error', dist_sync_on_step=False),
            'validation_accuracy': Accuracy('multiclass', num_classes=num_classes),
            'validation_loss': MeanMetric('error', dist_sync_on_step=False)
        }
    )
    return torch.compile(model)

def setup_mnist_cnn_model(num_classes: int, learning_rate: float, weight_decay:float, epochs: int, steps_per_epoch: int):
    model = Model(DEVICE, (
        nn.Conv2d(1, 8, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(8),
        nn.Conv2d(8, 16, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(16, 32, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Conv2d(32, 64, 3, 2, 1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    ))
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate * 4,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        anneal_strategy='cos'
    )
    model.configure(
        optimizer,
        scheduler,
        torch.nn.CrossEntropyLoss(),
        { 
            'accuracy': Accuracy('multiclass', num_classes=num_classes),
            'loss': MeanMetric('error', dist_sync_on_step=False),
            'validation_accuracy': Accuracy('multiclass', num_classes=num_classes),
            'validation_loss': MeanMetric('error', dist_sync_on_step=False)
        }
    )
    return torch.compile(model)
