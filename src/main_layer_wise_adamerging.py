import os

import time
import sys
import tqdm
sys.path.append('/root/workspace/arithmetic/ModelTinting')
import random
import math
import torch
from task_vectors import TaskVector
from eval import eval_single_dataset, eval_single_dataset_head, eval_single_dataset_preprocess_head, eval_single_dataset_preprocess_head_with_ece, eval_single_dataset_preprocess_mapping_head
from args import parse_arguments
from src.merging_cofficient import get_merging_cofficients
import datetime

now = str(datetime.datetime.now())
pth_save_path = f"/root/workspace/arithmetic/ModelTinting/pth_save_{now}"
if not os.path.exists(pth_save_path):
    os.makedirs(pth_save_path)

def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path+'/'+filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


############ 8 tasks ############
exam_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD'] # SUN397 | Cars | RESISC45 | EuroSAT | SVHN | GTSRB | MNIST | DTD
###################################


############ 14 tasks ############
# exam_datasets = [
#     "MNIST",
#     "Cars",
#     "DTD",
#     "EuroSAT",
#     "GTSRB",
#     "RESISC45",
#     "SUN397",
#     "SVHN",
#     "PCAM",
#     "CIFAR100",
#     "STL10",
#     "OxfordIIITPet",
#     "Flowers102",
#     "FER2013",
# ]
###################################



############ 20 tasks ############
# exam_datasets = [
#     "MNIST",
#     "Cars",
#     "DTD",
#     "EuroSAT",
#     "GTSRB",
#     "RESISC45",
#     "SUN397",
#     "SVHN",
#     "PCAM",
#     "CIFAR100",
#     "STL10",
#     "OxfordIIITPet",
#     "Flowers102",
#     "FER2013",
#     "CIFAR10",
#     "Food101",
#     "RenderedSST2",
#     "EMNIST",
#     "FashionMNIST",
#     "KMNIST",
# ]
###################################



args = parse_arguments()
model = args.model
print(f"Classifier training enabled: {args.classifier_train}")
print(args.losstype)


if args.twodataset == 'type1':
    exam_datasets = ['Cars', 'MNIST'] # SUN397 | Cars | RESISC45 | EuroSAT | SVHN | GTSRB | MNIST | DTD
    print(exam_datasets)

source_root_path = '/root/workspace/arithmetic/ModelTinting'
args.data_location = '/root/datasets/multitask'
args.model = model
args.save = source_root_path+'/checkpoints/' + model
args.logs_path = '/root/workspace/arithmetic/ModelTinting/logs/' + model
pretrained_checkpoint = source_root_path+'/checkpoints/'+model+'/zeroshot.pt'

str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
log = create_log_dir(args.logs_path, 'log_{}_Layer_wise_AdaMerging.txt'.format(str_time_))
args.log = log

task_vectors = [TaskVector(pretrained_checkpoint, source_root_path+'/checkpoints/'+model+'/'+dataset_name+'/finetuned.pt') for dataset_name in exam_datasets]



def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])

def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

def make_functional(mod):
    orig_params = tuple(mod.parameters())
    names = []
    for name, p in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
    return orig_params, names

def load_weights(mod, names, params):
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)

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
        prior = args.prior # default

        rlambdas = torch.ones(len(paramslist[0]), len(paramslist)-1) * prior  # (1 * tasks)
        if args.adastart:
            rlambdas = get_merging_cofficients('lw_adamerging', 'ViT-B-32')
            rlambdas = torch.tensor(rlambdas)[:,1:]
            
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.lambdas_raw = torch.nn.Parameter(rlambdas)

        if args.surgery:
            self.non_linear_func = torch.nn.ReLU()

        self.classifier = []
        for dataset_name in exam_datasets:

            if args.surgery:
                # mapping
                # ViT-B/32 and ViT-B/16
                down_proj = torch.nn.Linear(512, 16, bias=False)
                up_proj = torch.nn.Linear(16, 512, bias=False)
                # ViT-L/14
                # down_proj = torch.nn.Linear(768, 16, bias=False)
                # up_proj = torch.nn.Linear(16, 768, bias=False)
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


    def collect_trainable_params_group(self):
        param_groups = []
        if args.classifier_train:
            # Collect lambdas_raw parameters
            param_groups.append({'params': [self.lambdas_raw], 'lr': 1e-3})  # lambdas_raw lr
            
            # Collect classifier parameters
            for name in self.classifier:
                classifier_params = getattr(self, name).parameters()
                param_groups.append({'params': classifier_params, 'lr': 1e-2})  # classifier params lr

        return param_groups



    def collect_trainable_params_lam(self):
        return [self.lambdas_raw]

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
# paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in pretrained_model_dic.items())] # pretrain
# paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in tv.vector.items())  for i, tv in enumerate(task_vectors)] # task vectors
paramslist += [tuple(v.detach().requires_grad_().to(args.device) for _, v in pretrained_model_dic.items())] # pretrain
paramslist += [tuple(v.detach().requires_grad_().to(args.device) for _, v in tv.vector.items())  for i, tv in enumerate(task_vectors)] # task vectors

torch.cuda.empty_cache()
adamerging_mtl_model = AdaMerging(paramslist, model, names, exam_datasets)



adamerging_mtl_model = adamerging_mtl_model.to(args.device)
adamerging_mtl_model.pretrain_lambdas = adamerging_mtl_model.pretrain_lambdas.to(args.device)


# load trained classifier, merging coef  
# load_path = '/root/workspace/arithmetic/AdaMerging/pth_original/pth_save_new_ViT-B-32_crosspseudo_clsTrue_2024-10-01 07:13:34.427905/499.pth'
# load_path = '/root/workspace/arithmetic/AdaMerging/pth_save_new_ADAMERGING_14taskViT-B-32_selfentropy_clsFalse_prior0.1_noclampFalse_sparseFalse0.0001_clsinitrandFalse_adainitFalse_onlyclsFalse2024-10-30 17:10:13.124004/499.pth'
# load_path = '/root/workspace/arithmetic/AdaMerging/pth_save_new_ADAMERGING_20taskViT-B-32_selfentropy_clsFalse_prior0.1_noclampFalse_sparseFalse0.0001_clsinitrandFalse_adainitFalse_onlyclsFalse2024-10-31 02:41:43.984806/499.pth'
# load_st = torch.load(load_path)
# adamerging_mtl_model.load_state_dict(load_st, strict=False)



print('init lambda:')
print(adamerging_mtl_model.lambdas()[:,1])
print('collect_trainable_params:')
print(list(adamerging_mtl_model.collect_trainable_params()))
### adamerging_mtl_model.collect_trainable_params()[0] -> 158, 8


# # loss learnable coef
# loss_coefs = torch.nn.Parameter(torch.ones(8), requires_grad=True)



epochs = 500
if args.classifier_train:
    optimizer = torch.optim.Adam(adamerging_mtl_model.collect_trainable_params(), lr=1e-2, betas=(0.9, 0.999), weight_decay=0.)
else:
    optimizer = torch.optim.Adam(adamerging_mtl_model.collect_trainable_params(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.)
optimizer_lam = torch.optim.Adam(adamerging_mtl_model.collect_trainable_params_lam(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.)

# optimizer = torch.optim.Adam(adamerging_mtl_model.collect_trainable_params_group(), betas=(0.9, 0.999), weight_decay=0.)



from dataset.registry import get_dataset
from dataset.common import get_dataloader, maybe_dictionarize, get_dataloader_shuffle

Total_ACC = 0.
for dataset_name in exam_datasets:
    image_encoder = adamerging_mtl_model.get_image_encoder()
    classification_head = adamerging_mtl_model.get_classification_head(dataset_name)
    metrics = eval_single_dataset_preprocess_head(image_encoder, classification_head, dataset_name, args)
    Total_ACC += metrics['top1']
    log.info('Eval: init: ' + ' dataset: ' + str(dataset_name) + ' ACC: ' + str(metrics['top1']))
log.info('Eval: init: ' + ' Avg ACC:' + str(Total_ACC / len(exam_datasets)) + '\n')

Total_ACC = 0.
for i, dataset_name in enumerate(exam_datasets):
    image_encoder = task_vectors[i].apply_to(pretrained_checkpoint, scaling_coef=1.0)
    classification_head = adamerging_mtl_model.get_classification_head(dataset_name)
    metrics = eval_single_dataset_preprocess_head_with_ece(image_encoder, classification_head, dataset_name, args)
    Total_ACC += metrics['top1']
    log.info('Eval: ' + ' dataset: ' + str(dataset_name) + ' ACC: ' + str(metrics['top1']))
log.info('Eval: ' + ' Avg ACC:' + str(Total_ACC / len(exam_datasets)) + '\n')




epochs = 500

print('layerwise ece')


loss_func = torch.nn.L1Loss()
loss_func2 = torch.nn.MSELoss()
loss_func3 = torch.nn.SmoothL1Loss()


criterion = torch.nn.CrossEntropyLoss()

import torch.nn.functional as F

import pickle
cls_ft = {}
model_ft = {}
for dataset_name in exam_datasets:
    # individual model pseudo label 만들기위해
    finetune_model = args.model
    # finetuned = torch.load(source_root_path+'/checkpoints/'+finetune_model+'/'+dataset_name+'/finetuned.pt')
    try:
        finetuned = torch.load(source_root_path+'/checkpoints/'+finetune_model+'/'+dataset_name+'/finetuned.pt')
    except:
        finetuned = pickle.load(open(source_root_path+'/checkpoints/'+finetune_model+'/'+dataset_name+'/finetuned.pt', 'rb'))
    finetuned = finetuned
    classification_head_ft = get_classification_head(args, dataset_name)
    finetuned.eval()

    
    


    classification_head_ft.eval()
    print(classification_head_ft.weight)

    model_ft[dataset_name] = finetuned  
    cls_ft[dataset_name] = classification_head_ft


    if args.randominitclassifier:
        with torch.no_grad():
            torch.nn.init.xavier_uniform_(getattr(adamerging_mtl_model, f'classifier_{dataset_name}').weight)  # Xavier )
            torch.nn.init.constant_( (getattr(adamerging_mtl_model, f'classifier_{dataset_name}').bias), 0.0 ) # 0 )



    #######

Total_ACC = 0.
Total_ECE = 0.
for dataset_name in exam_datasets:
    image_encoder = adamerging_mtl_model.get_image_encoder()
    classification_head = adamerging_mtl_model.get_classification_head(dataset_name)

    if args.surgery:
        print('surgery!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        down_proj, up_proj = adamerging_mtl_model.get_feature_mapping_to_head(dataset_name)
        metrics = eval_single_dataset_preprocess_mapping_head(image_encoder, classification_head, dataset_name, args, down_proj, up_proj)
    else:
        metrics = eval_single_dataset_preprocess_head_with_ece(image_encoder, classification_head, dataset_name, args)
        Total_ECE += metrics['ece']

    Total_ACC += metrics['top1']


torch.save(adamerging_mtl_model.state_dict(), f'{pth_save_path}/_0.pth')



l1_loss = torch.nn.L1Loss()
smooth_l1_loss = torch.nn.SmoothL1Loss()
l2_loss = torch.nn.MSELoss()
def kl_divergence(p, q):
    return F.kl_div(p.log(), q, reduction='batchmean')

def js_divergence(p, q):
    m = (p + q) / 2
    return (kl_divergence(p, m) + kl_divergence(q, m)) / 2


for epoch in range(epochs):
    
    print(epoch)
    random.shuffle(exam_datasets)
    loss_coefs = torch.nn.functional.softmax(torch.randn(8), dim=-1)
    loss_cnt = 0
    for dataset_name in exam_datasets:
        losses = 0.
        optimizer.zero_grad()
        optimizer_lam.zero_grad()

        dataset = get_dataset(dataset_name, pretrained_model.val_preprocess, location=args.data_location, batch_size=32)  
        dataloader = get_dataloader_shuffle(dataset)
        
        loss_tmp = 0

        classification_head_ft = cls_ft[dataset_name]
        finetuned = model_ft[dataset_name]

        finetuned = finetuned.to(args.device)
        classification_head_ft = classification_head_ft.to(args.device)

        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data['images'].to(args.device)
            y = data['labels'].to(args.device)          


            outputs = adamerging_mtl_model(x, dataset_name)   
            if epoch < 0:
                loss = softmax_entropy(outputs).mean(0)
            else:
                if args.losstype == 'selfentropy':
                    loss = softmax_entropy(outputs).mean(0)   
                    finetuned = finetuned.to(args.device).cpu() 
                    classification_head_ft = classification_head_ft.cpu()
                    torch.cuda.empty_cache()

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

                elif args.losstype == 'crosspseudosoft':
                    with torch.no_grad():
                        finetuned_features = classification_head_ft(finetuned(x)).detach()
                    finetuned = finetuned.to(args.device).cpu() 
                    classification_head_ft = classification_head_ft.cpu()
                    torch.cuda.empty_cache()
                    pseudo_labels = finetuned_features.softmax(dim=-1)
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
                    print(cross1.sum().item(), cross2.sum().item(), self1.sum().item())


                    if cross2.sum().item() != 0:
                        print((cross1 + cross2).sum())
                        loss_cross = criterion(outputs[cross1 + cross2], ind_idx[cross1 + cross2])
                        loss = loss_cross
                        print('loss_cross', loss_cross)
                    else:
                        loss_cross1 = criterion(outputs[cross1], ind_idx[cross1])
                        loss = loss_cross1
                        print('loss_cross1', loss_cross1)
                
            loss_tmp += loss
            losses += loss

            if i == 0:  # Execute only one step   
                losses.backward()
                optimizer.step()
                break


    
    
    if ((epoch+1) % 50) == 0:

        

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
            



        if not os.path.exists(pth_save_path):
            os.makedirs(pth_save_path)

        torch.save(adamerging_mtl_model.state_dict(), f'{pth_save_path}/{epoch}.pth')

        log.info('Eval: Epoch: ' + str(epoch) + ' Avg ACC:' + str(Total_ACC / len(exam_datasets)) + '\n')

