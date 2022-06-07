from datasets import load_dataset
import transformers
from torch.utils.data import DataLoader
import torch
import numpy as np
from transformers import PerceiverFeatureExtractor, PerceiverPreTrainedModel
from transformers import AdamW
import tqdm
from datasets import load_metric
import sklearn
from sklearn.metrics import \
    accuracy_score, brier_score_loss, log_loss, \
    precision_score, recall_score
import sys
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

saving_name = sys.argv[1]

training = int(sys.argv[2])

# load cifar10 (only small portion for demonstration purposes)
train_ds, test_ds = load_dataset('cifar100', split=['train', 'test'])
# split up training into training + validation
splits = train_ds.train_test_split(test_size=0.1)
train_ds = splits['train']
val_ds = splits['test']

id2label = {idx: label for idx, label in enumerate(train_ds.features['fine_label'].names)}
label2id = {label: idx for idx, label in id2label.items()}


class _ECELoss(torch.nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = torch.nn.functional.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


feature_extractor = PerceiverFeatureExtractor()


def preprocess_images(examples):
    examples['pixel_values'] = feature_extractor(examples['img'], return_tensors="pt").pixel_values
    return examples


train_ds.set_transform(preprocess_images)
val_ds.set_transform(preprocess_images)
test_ds.set_transform(preprocess_images)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["fine_label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


train_batch_size = eval_batch_size = 64

train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=collate_fn, batch_size=train_batch_size)
val_dataloader = DataLoader(val_ds, collate_fn=collate_fn, batch_size=eval_batch_size)
test_dataloader = DataLoader(test_ds, collate_fn=collate_fn, batch_size=eval_batch_size)

batch = next(iter(train_dataloader))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import ViTForImageClassification
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                  num_labels=10,
                                                  id2label=id2label,
                                                  label2id=label2id)
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)#5e-6

model.train()
early_stopping = 5
count = 0


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


def evaluate(model, dataloader, stage, max_loss):
    model.eval()
    accuracy, brier, precision, recall, ece, logloss, loss_all = 0, 0, 0, 0, 0, 0, 0
    count_batch = 0
    for batch in tqdm.tqdm(dataloader):
        inputs = batch["pixel_values"].to(device)
        batch_size = len(batch["labels"])

        labels = batch["labels"].to(device)
        # forward pass
        logits = 0
        loss = 0
        ensemble_size = 1
        for i in range(ensemble_size):
            outputs = model(inputs, labels=labels)
            logits += outputs.logits.cpu().detach().numpy()
            loss += outputs.loss.cpu().detach().numpy()
        logits /= ensemble_size
        loss /= ensemble_size
        predictions = logits.argmax(-1)
        references = batch["labels"].numpy()
        loss = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(torch.from_numpy(logits)), batch["labels"])

        label = np.zeros_like(softmax(logits))
        label[np.arange(batch_size), references] = 1

        prob = softmax(logits)[np.arange(batch_size), references]
        accuracy += (references == predictions).sum()
        brier += brier_score_loss(np.ones_like(prob), prob) * eval_batch_size

        ece += _ECELoss()(torch.from_numpy(logits), torch.from_numpy(references)).item() * eval_batch_size

        count_batch += eval_batch_size
        loss_all += loss * eval_batch_size
    print(f"{stage} accuracy: { accuracy / count_batch }")
    print(f"{stage} brier: { brier / count_batch }")
    print(f"{stage} ece: { ece / count_batch }")
    print(f"{stage} loss: { loss_all / count_batch }")
    print(f"{stage} max_loss: { max_loss }")

    return model, loss_all, count_batch, max_loss

max_loss = float("inf")

if training:
    for epoch in range(1000):  # loop over the dataset multiple times
        model.train()
        print(f"Epoch {epoch}")
        size = len(train_dataloader.dataset)

        for i, batch in enumerate(tqdm.tqdm(train_dataloader)):
            inputs = batch["pixel_values"]
            inputs = inputs.to(device)
            labels = batch["labels"].to(device)

            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model, loss_all, count_batch, max_loss = evaluate(model, val_dataloader, "Val", max_loss)
        if (loss_all / count_batch < max_loss):
            max_loss = loss_all / count_batch
            count = 0
            torch.save(model.state_dict(), f"{saving_name}.pth")
        else:
            count += 1
            print("not smaller")

        if early_stopping == count:
            break

model.load_state_dict(torch.load(f'{saving_name}.pth'))

model, loss_all, count_batch, max_loss = evaluate(model, test_dataloader, "Test", max_loss)
