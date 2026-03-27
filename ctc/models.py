import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, MeanMetric
from tqdm import tqdm

from typing import Iterable

from traineval import DEVICE


def ctc_greedy_decode(logits, blank=2):
    predictions = logits.argmax(dim=2)
    T, B = predictions.shape
    decoded = []
    for b in range(B):
        prev = blank
        seq = []
        for t in range(T):
            p = int(predictions[t, b].item())
            if p != prev and p != blank:
                seq.append(p)
            prev = p
        decoded.append(seq)
    return decoded

def pad_batch_images(imgs, pad_value=0):
    def pad(img, w, h):
        nonlocal pad_value
        r = w - img.size(0)
        b = h - img.size(1)
        return F.pad(img, (0, r, 0, b), mode='constant', value=pad_value)

    max_w = max(map(lambda img: img.size(0) * img.size(2), imgs))
    max_h = max(map(lambda img: img.size(1), imgs))
    return torch.stack([pad(img, max_w, max_h) for img in imgs])


def collate(batch):
    imgs = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    lengths = [b[2] for b in batch]
    images_padded = pad_batch_images(imgs, pad_value=0)
    targets = torch.stack(targets).long()
    lenghts = torch.stack(lengths).long()
    return images_padded, targets, lenghts

#def levenshtein(a, b):
#    m, n = len(a), len(b)
#    dp = list(range(n+1))
#    for i in range(1, m + 1):
#        prev, dp[0] = dp[0], i
#        for j in range(1, n + 1):
#            cur = min(dp[j] + 1, dp[j - 1] + 1, prev + (0 if a[i - 1] == b[j - 1] else 1))
#            prev, dp[j] = dp[j], cur
#    return dp[n]


class CTCModel(nn.Module):
    def __init__(self, device, num_classes, backbone, rnn_in_dim, rnn_hidden, rnn_layers):
        super().__init__()
        self.backbone = backbone
        self.rnn = nn.LSTM(input_size=rnn_in_dim, hidden_size=rnn_hidden, num_layers=rnn_layers, batch_first=False, bidirectional=True)
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)
        self.backbone.to(device)
        self.to(device)
        self.device = device

    def forward(self, x):
        b, c, h, w = x.size()
        convoluted = self.backbone(x)
        _, c2, h2, w2 = convoluted.size()
        features = convoluted.permute(3, 0, 1, 2).contiguous()
        features = features.view(w2, b, h2 * c2)
        rnn_out, _ = self.rnn(features)
        logits = self.fc(rnn_out)
        return logits

    def configure(self, optimizer, scheduler, loss, metrics: dict):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.test_loss = metrics.get('loss').to(self.device)

    def fit(self, epochs: int, train_loader: DataLoader):
        self.train()

        logs = []

        for e in range(1, epochs + 1):

            if self.test_loss is not None:
                self.test_loss.reset()

            with tqdm(train_loader, unit='batch', desc=f'epoch {e}/{epochs}') as pbar:

                for images, targets, lengths in pbar:
                    images = images.to(self.device)
                    targets = targets.to(self.device)

                    self.optimizer.zero_grad()
                    logits = self(images)
                    log_probs = F.log_softmax(logits, dim=2)
                    loss = self.loss(log_probs, targets, lengths, lengths)
                    loss.backward()
                    self.optimizer.step()
                    if not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step()

                    self.test_loss.update(loss)

                    loss = None if self.test_loss is None else '{:.4f}'.format(self.test_loss.compute().item())
                    lr = None if self.optimizer is None else '{:.2e}'.format(self.optimizer.param_groups[0]['lr'])

                    pbar.set_postfix(loss=loss, lr=lr)

            log = {
                'loss': self.test_loss.compute().item()
            }
            logs.append(log)

    @torch.no_grad()
    def evaluate(self, test_loader):
        self.eval()

        self.test_loss.reset()

        for images, targets, lengths in test_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            logits = self(images)
            log_probs = F.log_softmax(logits, dim=2)
            loss = self.loss(log_probs, targets, lengths, lengths)
            
            self.test_loss.update(loss)

        return {
            'loss': self.test_loss.compute().item()
        }


def crnn_ctc_model(learning_rate: float, weight_decay: float):
    backbone = nn.ModuleList((
        nn.Conv2d(1, 64, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Conv2d(64, 128, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(128, 256, 3, 1, 1),
    ))
    model = CTCModel(DEVICE, 3, backbone, 256, 256, 2)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0)
    model.configure(
        optimizer,
        scheduler,
        nn.CTCLoss(blank=2, reduction='mean'),
        { 
            'loss': MeanMetric('error', dist_sync_on_step=False),
        }
    )
    return torch.compile(model)
