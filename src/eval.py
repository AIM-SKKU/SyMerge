import os
import json
import tqdm

import torch
import numpy as np

import utils

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


from dataset.common import get_dataloader, maybe_dictionarize
from heads import get_classification_head
from modeling import ImageClassifier, ImageClassifierWithMapping

from dataset.registry import get_dataset

def eval_single_dataset(image_encoder, dataset_name, args):
    classification_head = get_classification_head(args, dataset_name)
    model = ImageClassifier(image_encoder, classification_head)

    model.eval()

    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size
    )
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=None)
    device = args.device

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)

            logits = utils.get_logits(x, model)

            pred = logits.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()
            
            n += y.size(0)

        top1 = correct / n

    metrics = {'top1': top1}
    print(f'Done evaluating on {dataset_name}. Accuracy: {100*top1:.2f}%')
    
    return metrics

#  for ece calc
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

def eval_single_dataset_with_ece(image_encoder, dataset_name, args):
    classification_head = get_classification_head(args, dataset_name)
    model = ImageClassifier(image_encoder, classification_head)

    model.eval()

    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size
    )
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=None)
    device = args.device

    ece_criterion = _ECELoss().cuda()
    logits_list =[]
    labels_list = []
    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)

            logits = utils.get_logits(x, model)
            logits_list.append(logits)
            labels_list.append(y)

            pred = logits.argmax(dim=1, keepdim=True).to(device)

            

            correct += pred.eq(y.view_as(pred)).sum().item()
            
            n += y.size(0)

        top1 = correct / n

    logits_list = torch.cat(logits_list).cuda()
    labels_list = torch.cat(labels_list).cuda()
    ece = ece_criterion(logits_list, labels_list).item()
    metrics = {'top1': top1}
    print(f'Done evaluating on {dataset_name}. Accuracy: {100*top1:.2f}%')
    print(f'ece : {100*ece:.2f}%')
    return metrics
###################################################



def eval_single_dataset_head(image_encoder, head, dataset_name, args):
    model = ImageClassifier(image_encoder, head)

    model.eval()

    dataset = get_dataset(dataset_name, model.val_preprocess, location=args.data_location,  batch_size=args.batch_size)
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
    device = args.device

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)

            logits = utils.get_logits(x, model)

            pred = logits.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()

            n += y.size(0)

        top1 = correct / n

    metrics = {'top1': top1}
    print(f'Done evaluating on {dataset_name}. Accuracy: {100 * top1:.2f}%')

    return metrics

def eval_single_dataset_preprocess_head(image_encoder, head, dataset_name, args):
    model = ImageClassifier(image_encoder, head)

    model.eval()

    dataset = get_dataset(dataset_name, model.val_preprocess, location=args.data_location,  batch_size=args.batch_size)
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
    device = args.device

    print(args.device)
    print(next(model.parameters()).is_cuda)

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)

            logits = utils.get_logits(x, model)

            pred = logits.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()

            n += y.size(0)

        top1 = correct / n

    metrics = {'top1': top1}
   
    return metrics

# ece
def eval_single_dataset_preprocess_head_with_ece(image_encoder, head, dataset_name, args):
    model = ImageClassifier(image_encoder, head)

    model.eval()

    dataset = get_dataset(dataset_name, model.val_preprocess, location=args.data_location,  batch_size=args.batch_size)
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
    device = args.device

    print(args.device)
    print(next(model.parameters()).is_cuda)
    ece_criterion = _ECELoss().cuda()
    logits_list =[]
    labels_list = []
    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)

            logits = utils.get_logits(x, model)
            logits_list.append(logits)
            labels_list.append(y)

            pred = logits.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()

            n += y.size(0)

        top1 = correct / n
    logits_list = torch.cat(logits_list).cuda()
    labels_list = torch.cat(labels_list).cuda()
    ece = ece_criterion(logits_list, labels_list).item()

    metrics = {'top1': top1, 'ece':ece}
    print(f'Done evaluating on {dataset_name}. Accuracy: {100 * top1:.2f}%')
    print(f'ECE : {100*ece:.2f}%')
    return metrics



def evaluate(image_encoder, args):
    if args.eval_datasets is None:
        return
    info = vars(args)
    for i, dataset_name in enumerate(args.eval_datasets):
        print('Evaluating on', dataset_name)

        results = eval_single_dataset(image_encoder, dataset_name, args)

        if 'top1' in results:
            print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
        for key, val in results.items():
            if 'worst' in key or 'f1' in key.lower() or 'pm0' in key:
                print(f"{dataset_name} {key}: {val:.4f}")
            info[dataset_name + ':' + key] = val

    if args.results_db is not None:
        dirname = os.path.dirname(args.results_db)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(args.results_db, 'a+') as f:
            f.write(json.dumps(info) + '\n')
        print(f'Results saved to {args.results_db}.')
    else:
        print('Results not saved (to do so, use --results_db to specify a path).')

    return info





def eval_single_dataset_preprocess_mapping_head(image_encoder, head, dataset_name, args, down_proj, up_proj):
    model = ImageClassifierWithMapping(image_encoder, head, down_proj, up_proj)
    model.eval()
    dataset = get_dataset(dataset_name, model.val_preprocess, location=args.data_location,  batch_size=args.batch_size)
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
    device = args.device

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)

            logits = utils.get_logits(x, model)

            pred = logits.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()

            n += y.size(0)

        top1 = correct / n

    metrics = {'top1': top1}
    print(f'Done evaluating on {dataset_name}. Accuracy: {100 * top1:.2f}%')

    return metrics


def eval_compare_individual_merged(image_encoder, head, dataset_name, args, finetuned, head_ft):
    model = ImageClassifier(image_encoder, head)
    finetuned_model = ImageClassifier(finetuned, head_ft)

    model.eval()
    finetuned_model.eval()

    dataset = get_dataset(dataset_name, model.val_preprocess, location=args.data_location,  batch_size=args.batch_size)
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
    device = args.device

    print(args.device)
    print(next(model.parameters()).is_cuda)
    ece_criterion = _ECELoss().cuda()
    logits_list =[]
    labels_list = []

    model_correct_but_finetuned_wrong = 0
    tmp = []

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)

            logits = utils.get_logits(x, model)
            logits_finetuned = utils.get_logits(x, finetuned_model)

            logits_list.append(logits)
            labels_list.append(y)

            pred = logits.argmax(dim=1, keepdim=True).to(device)
            pred_finetuned = logits_finetuned.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()

            model_correct = pred.eq(y.view_as(pred)).squeeze()
            finetuned_wrong = ~pred_finetuned.eq(y.view_as(pred_finetuned)).squeeze()

            model_wrong = ~pred.eq(y.view_as(pred)).squeeze()
            finetuned_correct = pred_finetuned.eq(y.view_as(pred_finetuned)).squeeze()

            model_correct_but_finetuned_wrong += (model_wrong & finetuned_correct).sum().item()
            # tmp.append(x[model_wrong & finetuned_correct])

            n += y.size(0)


        top1 = correct / n
    
    # img_icfw = torch.cat(tmp, dim=0)

    # if img_icfw.shape[0] > 50:
    #     indices = torch.randperm(img_icfw.shape[0])[:50]  # 0번째 차원에서 랜덤한 인덱스 50개 선택
    #     img_icfw = img_icfw[indices]  # 선택한 인덱스를 기반으로 새로운 텐서 생성

    # for i in range(img_icfw.shape[0]):
    #     visualize_images(img_icfw[i], i, dataset_name)
        

    logits_list = torch.cat(logits_list).cuda()
    labels_list = torch.cat(labels_list).cuda()
    ece = ece_criterion(logits_list, labels_list).item()

    metrics = {'top1': top1, 'ece':ece}
    print(f'Done evaluating on {dataset_name}. Accuracy: {100 * top1:.2f}%')
    print(f'ECE : {100*ece:.2f}%')
    return metrics, model_correct_but_finetuned_wrong




def eval_single_dataset_preprocess_mapping_head_compare(image_encoder, head, dataset_name, args, down_proj, up_proj, finetuned, head_ft):
    # model = ImageClassifier(image_encoder, head)
    model = ImageClassifierWithMapping(image_encoder, head, down_proj, up_proj)
    model.eval()
    finetuned_model = ImageClassifier(finetuned, head_ft)
    finetuned_model.eval()

    dataset = get_dataset(dataset_name, model.val_preprocess, location=args.data_location,  batch_size=args.batch_size)
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
    device = args.device

    model_correct_but_finetuned_wrong = 0
    model_wrong_but_finetuned_correct = 0
    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)

            logits = utils.get_logits(x, model)
            logits_finetuned = utils.get_logits(x, finetuned_model)

            pred = logits.argmax(dim=1, keepdim=True).to(device)
            pred_finetuned = logits_finetuned.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()

            model_correct = pred.eq(y.view_as(pred)).squeeze()
            finetuned_wrong = ~pred_finetuned.eq(y.view_as(pred_finetuned)).squeeze()

            model_wrong = ~pred.eq(y.view_as(pred)).squeeze()
            finetuned_correct = pred_finetuned.eq(y.view_as(pred_finetuned)).squeeze()

            model_correct_but_finetuned_wrong += (model_correct & finetuned_wrong).sum().item()

            model_wrong_but_finetuned_correct += (model_wrong & finetuned_correct).sum().item()

            n += y.size(0)

        top1 = correct / n

    metrics = {'top1': top1}
    print(f'Done evaluating on {dataset_name}. Accuracy: {100 * top1:.2f}%')
    print('model_correct_but_finetuned_wrong', model_correct_but_finetuned_wrong)
    print('model_wrong_but_finetuned_correct', model_wrong_but_finetuned_correct)
    return metrics,  model_correct_but_finetuned_wrong, model_wrong_but_finetuned_correct