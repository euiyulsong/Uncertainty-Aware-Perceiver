from datasets import load_dataset
import transformers
from torch.utils.data import DataLoader
import torch
import numpy as np
from transformers import PerceiverFeatureExtractor, PerceiverPreTrainedModel
from transformers import AdamW
import tqdm
import scipy
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


class CustomPerceiverForImageClassificationLearned(PerceiverPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        trainable_position_encoding_kwargs_preprocessor = dict(num_channels=256, index_dims=config.image_size ** 2)
        trainable_position_encoding_kwargs_decoder = dict(num_channels=config.d_latents, index_dims=1)

        self.num_labels = config.num_labels
        self.perceiver = transformers.PerceiverModel(
            config,
            input_preprocessor=transformers.models.perceiver.modeling_perceiver.PerceiverImagePreprocessor(
                config,
                prep_type="conv1x1",
                spatial_downsample=1,
                out_channels=256,
                position_encoding_type="trainable",
                concat_or_add_pos="concat",
                project_pos_dim=256,
                trainable_position_encoding_kwargs=trainable_position_encoding_kwargs_preprocessor,
            ),
            decoder=transformers.models.perceiver.modeling_perceiver.PerceiverClassificationDecoder(
                config,
                num_channels=config.d_latents,
                trainable_position_encoding_kwargs=trainable_position_encoding_kwargs_decoder,
                use_query_residual=True,
            ),
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            inputs: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            labels: Optional[torch.Tensor] = None,
            return_dict: Optional[bool] = None,
            pixel_values: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, transformers.models.perceiver.modeling_perceiver.PerceiverClassifierOutput]:

        if inputs is not None and pixel_values is not None:
            raise ValueError("You cannot use both `inputs` and `pixel_values`")
        elif inputs is None and pixel_values is not None:
            inputs = pixel_values

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.perceiver(
            inputs=inputs,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits if return_dict else outputs[0]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return transformers.models.perceiver.modeling_perceiver.PerceiverClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions)


# load cifar10 (only small portion for demonstration purposes)
train_ds, test_ds = load_dataset('cifar10', split=['train', 'test'])
# split up training into training + validation
splits = train_ds.train_test_split(test_size=0.1)
train_ds = splits['train']
val_ds = splits['test']

id2label = {idx: label for idx, label in enumerate(train_ds.features['label'].names)}
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
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


train_batch_size = eval_batch_size = 4

train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=collate_fn, batch_size=train_batch_size)
val_dataloader = DataLoader(val_ds, collate_fn=collate_fn, batch_size=eval_batch_size)
test_dataloader = DataLoader(test_ds, collate_fn=collate_fn, batch_size=eval_batch_size)

batch = next(iter(train_dataloader))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = transformers.PerceiverForImageClassificationFourier.from_pretrained("deepmind/vision-perceiver-fourier",
                                                                     num_labels=10,
                                                                     id2label=id2label,
                                                                     label2id=label2id,
                                                                     ignore_mismatched_sizes=True)

model2 = transformers.PerceiverForImageClassificationFourier.from_pretrained("deepmind/vision-perceiver-fourier",
                                                                     num_labels=10,
                                                                     id2label=id2label,
                                                                     label2id=label2id,
                                                                     ignore_mismatched_sizes=True)
model3 = transformers.PerceiverForImageClassificationFourier.from_pretrained("deepmind/vision-perceiver-fourier",
                                                                     num_labels=10,
                                                                     id2label=id2label,
                                                                     label2id=label2id,
                                                                     ignore_mismatched_sizes=True)
model4 = transformers.PerceiverForImageClassificationFourier.from_pretrained("deepmind/vision-perceiver-fourier",
                                                                     num_labels=10,
                                                                     id2label=id2label,
                                                                     label2id=label2id,
                                                                     ignore_mismatched_sizes=True)
model.load_state_dict(torch.load(f'perceiver_fourier_10_2.pth'))
model2.load_state_dict(torch.load(f'perceiver_fourier_10_3.pth'))
model3.load_state_dict(torch.load(f'perceiver_fourier_10_4.pth'))
model4.load_state_dict(torch.load(f'perceiver_fourier_10.pth'))

class MyEnsemble(torch.nn.Module):
    def __init__(self, modelA, modelB, modelC, modelD):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.modelD = modelD
        self.temperature = torch.nn.Parameter(torch.ones(1) * 0.898)

    def temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def get_optimal_temperature(self,
            confidences: torch.Tensor,
            true_labels: torch.Tensor,
    ) -> float:
        def obj(t):
            target = true_labels.numpy()
            return -np.log(
                1e-12 + np.exp(torch.log_softmax(
                    torch.Tensor(
                        confidences.log().numpy() / t
                    ), dim=1
                ).data.numpy()
                               )[np.arange(len(target)), target]).mean()

        return scipy.optimize.minimize(
            obj, 1.0, method="nelder-mead", options={"xtol": 1e-3}
        ).x[0]

    def forward(self, xx, target):
        m = torch.nn.Softmax(dim=1)
        xx = xx.to(device)
        self.modelA = self.modelA.to(device)
        x1, x2, x3, x4 = 0, 0, 0, 0
        ensemble = 1
        for i in range(ensemble):
            x1 += self.modelA(xx).logits.cpu().detach()
        x1 /= ensemble
        self.modelA = self.modelA.to("cpu")
        self.modelB = self.modelB.to(device)
        for i in range(ensemble):
            x2 += self.modelB(xx).logits.cpu().detach()
        x2 /= ensemble
        self.modelB = self.modelB.to("cpu")
        self.modelC = self.modelC.to(device)
        for i in range(ensemble):
            x3 += self.modelC(xx).logits.cpu().detach()
        x3 /= ensemble
        self.modelC = self.modelC.to("cpu")
        self.modelD = self.modelD.to(device)
        for i in range(ensemble):
            x4 += self.modelD(xx).logits.cpu().detach()
        x4 /= ensemble
        self.modelD = self.modelD.to("cpu")
        x1_s = m(x1)
        x2_s = m(x2)
        x3_s = m(x3)
        x4_s = m(x4)

        optimizer1 = self.get_optimal_temperature(x1_s, target)
        optimizer2 = self.get_optimal_temperature(x2_s, target)
        optimizer3 = self.get_optimal_temperature(x3_s, target)
        optimizer4 = self.get_optimal_temperature(x4_s, target)
        x1_s = m(x1 / optimizer1)
        x2_s = m(x2 / optimizer2)
        x3_s = m(x3 / optimizer3)
        x4_s = m(x4 / optimizer4)
        x = torch.mean(torch.stack([x1_s, x2_s, x3_s, x4_s]), 0)
        return x, x1 / optimizer1, x2 / optimizer2, x3 / optimizer3, x4 / optimizer4


net = MyEnsemble(model, model2, model3, model4)


optimizer = AdamW(model.parameters(), lr=5e-6)

early_stopping = 15
count = 0


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


def evaluate(model, dataloader, stage):
    model.eval()
    test_loss = 0.0
    correct = 0.0
    test_bs = 0.0
    idx = 0
    ece_all = 0
    for i, batch in enumerate(tqdm.tqdm(dataloader)):
        images = batch["pixel_values"]
        labels = batch["labels"]
        batch_size = len(batch["labels"])

        outputs, x1_s, x2_s, x3_s, x4_s = net(images, labels)

        labels = labels
        max = torch.max(
                outputs, dim=1
            )[1]

        acc = max.eq(labels).float().mean().item()

        loss = torch.nn.functional.nll_loss(
            torch.log(outputs), labels
        ).item()
        loss_ece = (_ECELoss()(x1_s, labels) + _ECELoss()(x2_s, labels) + _ECELoss()(x3_s, labels) + _ECELoss()(x4_s, labels)) / 4
        ece_all += loss_ece.item() * batch_size

        targets = torch.eye(
            outputs.size(1)
        )[labels].long()
        targets = targets

        bs = torch.mean(
            torch.sum((outputs - targets)**2, dim=1)
        ).item()
        test_bs += bs * batch_size
        test_loss += loss * batch_size
        _, preds = outputs.max(1)

        correct += acc * batch_size

        print(f'Test set: Epoch: {i}, Average loss: {loss}, Accuracy: {acc}, Brier Score:{bs}, ECE:{loss_ece.item()}')
        idx += batch_size

    print(f'Test set: Average loss: {test_loss / idx }, Accuracy: {correct/ idx }, Brier Score:{test_bs / idx }, ECE:{ece_all / idx }')

evaluate(net, test_dataloader, "Test")
