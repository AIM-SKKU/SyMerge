import os

import time
import sys
sys.path.append('../')

import tqdm
import random
import math
import torch
from task_vectors import TaskVector
from eval import eval_single_dataset_preprocess_head_with_ece, eval_single_dataset_preprocess_mapping_head
from args import parse_arguments
from src.merging_cofficient import get_merging_cofficients
from utils import make_functional, load_weights, create_log_dir
from torch.cuda.amp import autocast, GradScaler
import pickle
from dataset.registry import get_dataset
from dataset.common import get_dataloader, maybe_dictionarize, get_dataloader_shuffle

args = parse_arguments()
if args.config:
    import yaml
    from pathlib import Path

    with Path(args.config).expanduser().open("r") as f:
        cfg = yaml.safe_load(f) or {}
    for key, value in cfg.items():
        setattr(args, key, value)

    if isinstance(args.exam_datasets, str):
        args.exam_datasets = args.exam_datasets.split(",")


model = args.model
exam_datasets = args.exam_datasets
data_dir = args.data_location
batch_size = args.batch_size
epochs = args.epochs

use_amp = args.use_amp
scaler = GradScaler(enabled=use_amp)


checkpoints_root = Path(args.checkpoints_root)
model_root = checkpoints_root / args.model
pretrained_checkpoint = str(model_root / args.pretrained_checkpoint)
args.save = str(model_root)
args.logs_path = str(Path(args.logs_path) / args.model)

str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
log = create_log_dir(args.logs_path, 'log_{}_SyMerge.txt'.format(str_time_))

task_vectors = [TaskVector(pretrained_checkpoint, str(model_root / dataset_name / 'finetuned.pt')) for dataset_name in exam_datasets]

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, initial_weights=None):
        super(ModelWrapper, self).__init__()
        self.model = model

        if hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        features = self.model(images)
        return features

from heads import get_classification_head
class AdaMerging(torch.nn.Module):
    def __init__(self, paramslist, model, names, exam_datasets):
        super(AdaMerging, self).__init__()
        self.paramslist = paramslist
        self.model = model
        self.names = names
        self.feature = None 
        self.pretrain_lambdas = torch.ones(len(paramslist[0]), 1)
        prior = args.prior

        rlambdas = torch.ones(len(paramslist[0]), len(paramslist)-1) * prior  # (1 * tasks)
        if args.adastart:
            rlambdas = get_merging_cofficients('lw_adamerging', 'ViT-B-32')
            rlambdas = torch.tensor(rlambdas)[:,1:]
            
        self.lambdas_raw = torch.nn.Parameter(rlambdas)

        if args.surgery:
            self.non_linear_func = torch.nn.ReLU()

        self.classifier = []
        for dataset_name in exam_datasets:

            if args.surgery:
                # mapping
                # ViT-B/32 and ViT-B/16
                if args.model != 'ViT-L-14':
                    down_proj = torch.nn.Linear(512, 16, bias=False)
                    up_proj = torch.nn.Linear(16, 512, bias=False)
                # ViT-L/14
                else:
                    down_proj = torch.nn.Linear(768, 16, bias=False)
                    up_proj = torch.nn.Linear(16, 768, bias=False)
                torch.nn.init.kaiming_uniform_(down_proj.weight, a=math.sqrt(5))
                torch.nn.init.zeros_(up_proj.weight)
                self.add_module('feature_mapping_to_head_down_proj_{}'.format(dataset_name), down_proj.to(args.device))
                self.add_module('feature_mapping_to_head_up_proj_{}'.format(dataset_name), up_proj.to(args.device))


            classification_head = get_classification_head(args, dataset_name)
            layer_name = 'classifier_{}'.format(dataset_name)
            self.add_module(layer_name, classification_head.to(args.device))
            self.classifier.append(layer_name)


    def lambdas(self):
        task_lambdas = torch.clamp(self.lambdas_raw, min=0.0, max=1.0)
        if args.noclamp:
            task_lambdas = self.lambdas_raw
        lambdass = torch.cat((self.pretrain_lambdas, task_lambdas), 1)
        return lambdass


    def collect_trainable_params(self):
        if args.classifier_train:
            # Collect lambdas_raw and classifier parameters
            params = [self.lambdas_raw]   # 158, 8
            if args.onlyclassifiertrain:
                params = []
            for name in self.classifier:
                classifier_params = getattr(self, name).parameters()
                params.extend(classifier_params)
            
                if args.surgery:
                    down_proj = getattr(self, 'feature_mapping_to_head_down_proj_{}'.format(name[11:]))
                    up_proj = getattr(self, 'feature_mapping_to_head_up_proj_{}'.format(name[11:]))
                    params.append(down_proj.weight)
                    params.append(up_proj.weight)

            return params
            
        else:
            params = [self.lambdas_raw]
            
            if args.surgery:
                params = []
                for name in self.classifier:
                    down_proj = getattr(self, 'feature_mapping_to_head_down_proj_{}'.format(name[11:]))
                    up_proj = getattr(self, 'feature_mapping_to_head_up_proj_{}'.format(name[11:]))
                    params.append(down_proj.weight)
                    params.append(up_proj.weight)
            return params



    def get_classification_head(self, dataset_name):
        layer_name = 'classifier_{}'.format(dataset_name)
        classification_head = getattr(self, layer_name)
        return classification_head

    def get_image_encoder(self):
        alph = self.lambdas()
        params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[j].cpu()))) for j, p in enumerate(zip(*self.paramslist)))
        params = tuple(p.cuda(0) for p in params)
        load_weights(self.model, self.names, params)
        return self.model
    
    def get_feature_mapping_to_head(self, dataset_name):
        down_proj = getattr(self, 'feature_mapping_to_head_down_proj_{}'.format(dataset_name))
        up_proj = getattr(self, 'feature_mapping_to_head_up_proj_{}'.format(dataset_name))
        return down_proj, up_proj
    
    def forward(self, inp, dataset_name):
        alph = self.lambdas()   # alph.shape 158, 9
        params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[j].cpu()))) for j, p in enumerate(zip(*self.paramslist)))   # len 158

        params = tuple(p.cuda(0) for p in params)
        load_weights(self.model, self.names, params)
        feature = self.model(inp)   # feature : task, 512 

        self.feature = feature

        if args.surgery:
            feature0 = feature

            # feature bias
            down_proj = getattr(self, 'feature_mapping_to_head_down_proj_{}'.format(dataset_name))
            up_proj = getattr(self, 'feature_mapping_to_head_up_proj_{}'.format(dataset_name))
            feature_sub = down_proj(feature)
            feature_sub = self.non_linear_func(feature_sub)
            feature_sub = up_proj(feature_sub)

            # surgery feature
            feature = feature0 - feature_sub

            self.feature = feature


        layer_name = 'classifier_{}'.format(dataset_name)
        classification_head = getattr(self, layer_name)
        out = classification_head(feature)
        return out

def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)



pretrained_model = torch.load(pretrained_checkpoint)
pretrained_model_dic = pretrained_model.state_dict()

model = ModelWrapper(pretrained_model, exam_datasets)
model = model.to(args.device)
_, names = make_functional(model)

paramslist = []
paramslist += [tuple(v.detach().to(args.device) for _, v in pretrained_model_dic.items())] # pretrain
paramslist += [tuple(v.detach().to(args.device) for _, v in tv.vector.items())  for i, tv in enumerate(task_vectors)] # task vectors

adamerging_mtl_model = AdaMerging(paramslist, model, names, exam_datasets)



adamerging_mtl_model = adamerging_mtl_model.to(args.device)
adamerging_mtl_model.pretrain_lambdas = adamerging_mtl_model.pretrain_lambdas.to(args.device)



if args.classifier_train:
    optimizer = torch.optim.Adam(adamerging_mtl_model.collect_trainable_params(), lr=1e-2, betas=(0.9, 0.999), weight_decay=0.)
else:
    optimizer = torch.optim.Adam(adamerging_mtl_model.collect_trainable_params(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.)


criterion = torch.nn.CrossEntropyLoss()
loss_func = torch.nn.L1Loss()

cls_ft = {}
model_ft = {}
for dataset_name in exam_datasets:
    finetune_model = args.model
    try:
        finetuned = torch.load(str(model_root / dataset_name / 'finetuned.pt'))
    except:
        finetuned = pickle.load(open(str(model_root / dataset_name / 'finetuned.pt'), 'rb'))
    finetuned = finetuned
    classification_head_ft = get_classification_head(args, dataset_name)
    finetuned.eval()

    classification_head_ft.eval()
    model_ft[dataset_name] = finetuned  
    cls_ft[dataset_name] = classification_head_ft

    if args.randominitclassifier:
        with torch.no_grad():
            torch.nn.init.xavier_uniform_(getattr(adamerging_mtl_model, f'classifier_{dataset_name}').weight)  # Xavier )
            torch.nn.init.constant_( (getattr(adamerging_mtl_model, f'classifier_{dataset_name}').bias), 0.0 ) # 0 )

# Init Evaluation
# Total_ACC = 0.
# Total_ECE = 0.
# for dataset_name in exam_datasets:
#     image_encoder = adamerging_mtl_model.get_image_encoder()
#     classification_head = adamerging_mtl_model.get_classification_head(dataset_name)

#     if args.surgery:
#         down_proj, up_proj = adamerging_mtl_model.get_feature_mapping_to_head(dataset_name)
#         metrics = eval_single_dataset_preprocess_mapping_head(image_encoder, classification_head, dataset_name, args, down_proj, up_proj)
#     else:
#         metrics = eval_single_dataset_preprocess_head_with_ece(image_encoder, classification_head, dataset_name, args)
#         Total_ECE += metrics['ece']

#     Total_ACC += metrics['top1']

print("Creating dataloaders...")
dataloaders = {
    name: get_dataloader_shuffle(
        get_dataset(name, pretrained_model.val_preprocess, location=data_dir, batch_size=batch_size)
    )
    for name in exam_datasets
}

num_steps = max(len(loader) for loader in dataloaders.values())
print(f"Total steps per epoch: {num_steps}")

iterators = {name: iter(loader) for name, loader in dataloaders.items()}



for epoch in range(epochs):
    
    print(epoch)
    random.shuffle(exam_datasets)

    for dataset_name in exam_datasets:
        losses = 0.
        optimizer.zero_grad()

        

        classification_head_ft = cls_ft[dataset_name]
        finetuned = model_ft[dataset_name]

        finetuned = finetuned.to(args.device)
        classification_head_ft = classification_head_ft.to(args.device)

        try:
            data = next(iterators[dataset_name])
        except StopIteration:
            print(f"\nRe-initializing iterator for {dataset_name}")
            iterators[dataset_name] = iter(dataloaders[dataset_name])
            data = next(iterators[dataset_name])
        data = maybe_dictionarize(data)
        x = data['images'].to(args.device)
        y = data['labels'].to(args.device)    


        with autocast():

            outputs = adamerging_mtl_model(x, dataset_name)   
            if epoch < 0:
                loss = softmax_entropy(outputs).mean(0)
            else:
                if args.losstype == 'selfentropy':
                    loss = softmax_entropy(outputs).mean(0)   
                    finetuned = finetuned.to(args.device).cpu() 
                    classification_head_ft = classification_head_ft.cpu()
                    torch.cuda.empty_cache()
                
                elif args.losstype == 'l1feature':
                    outputs_feat = adamerging_mtl_model.feature
                    finetuned_features = finetuned(x).detach()   # image encoder feature
                    loss = loss_func(outputs_feat, finetuned_features)
                    
                    finetuned = finetuned.to(args.device).cpu() 
                    classification_head_ft = classification_head_ft.cpu()

                elif args.losstype == 'crossgt':
                    loss = criterion(outputs, y)

                elif args.losstype == 'crosspseudo':
                    with torch.no_grad():
                        finetuned_features = classification_head_ft(finetuned(x)).detach()
                        finetuned = finetuned.to(args.device).cpu() 
                        classification_head_ft = classification_head_ft.cpu()
                        torch.cuda.empty_cache()

                    _, pseudo_labels = torch.max(finetuned_features, 1)
                    loss = criterion(outputs, pseudo_labels)     


                elif args.losstype == 'crosspseudoconf':
                    with torch.no_grad():
                        finetuned_features = classification_head_ft(finetuned(x)).detach()
                    finetuned = finetuned.to(args.device).cpu() 
                    classification_head_ft = classification_head_ft.cpu()
                    torch.cuda.empty_cache()

                    ind_prob = finetuned_features.softmax(dim=-1)   
                    out_prob = outputs.softmax(dim=-1)   

                    ind_value, ind_idx  = torch.max(ind_prob, 1)  
                    out_value, out_idx = torch.max(out_prob, 1)

                    cross1 = (ind_idx == out_idx)
                    cross2 = (ind_idx != out_idx) & (ind_value >= out_value)   
                    self1 = (ind_idx != out_idx) & (ind_value < out_value)  


                    if cross2.sum().item() != 0:
                        loss_cross = criterion(outputs[cross1 + cross2], ind_idx[cross1 + cross2])
                        loss = loss_cross
                    else:
                        loss_cross1 = criterion(outputs[cross1], ind_idx[cross1])
                        loss = loss_cross1
                
            losses += loss

            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()


    
    
    if ((epoch+1) % 500) == 0:

        

        Total_ACC = 0.
        Total_ECE = 0.
        for dataset_name in exam_datasets:
            image_encoder = adamerging_mtl_model.get_image_encoder()
            classification_head = adamerging_mtl_model.get_classification_head(dataset_name)
            
            if args.surgery:
                down_proj, up_proj = adamerging_mtl_model.get_feature_mapping_to_head(dataset_name)
                metrics = eval_single_dataset_preprocess_mapping_head(image_encoder, classification_head, dataset_name, args, down_proj, up_proj)
            else:
                metrics = eval_single_dataset_preprocess_head_with_ece(image_encoder, classification_head, dataset_name, args)
                Total_ECE += metrics['ece']
                log.info('Eval: Epoch: ' + str(epoch) + ' dataset: ' + str(dataset_name) + ' ece: ' + str(metrics['ece']))

            Total_ACC += metrics['top1']
            
            log.info('Eval: Epoch: ' + str(epoch) + ' dataset: ' + str(dataset_name) + ' ACC: ' + str(metrics['top1']))
            


        log.info('Eval: Epoch: ' + str(epoch) + ' Avg ACC:' + str(Total_ACC / len(exam_datasets)) + '\n')

