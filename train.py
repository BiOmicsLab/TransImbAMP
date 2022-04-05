import os, time, argparse, math
import optparse
from optparse import OptionParser
from warnings import warn
from typing import Union
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc, matthews_corrcoef, f1_score, hamming_loss
from sklearn.utils import shuffle
from data import AMPDataset
from loss import AsymmetricLossOptimized, ASLSingleLabel
from model import BERTAMP


def train(dataloader, model, criterion, opt, epoch=None, use_cuda=True):
    """Standard training function.

    Executing a training process within an epoch.

    Args:
        dataloader: pytorch DataLoader for handling training.
        model: nn.Module to train.
        criterion: nn.criterion for the loss function of training.
        opt: torch.optim.optimizer for optimization of traning.
        epoch: indicate the specific epoch at this training.
        use_cuda: whether use cuda.

    Returns:
        The avaerage loss of this training step

        total_loss / cnt
    """
    model.train()
    total_loss = 0
    cnt = 0
    desc = "Training (Epoch {:d})".format(epoch) if epoch is not None else "Training"
    for i, batch_items in tqdm(enumerate(dataloader), total=dataloader.__len__(), leave=False, desc=desc):
        input_ids = batch_items['input_ids']
        input_mask = batch_items['input_mask']
        trg = batch_items['targets']
        if use_cuda:
            input_ids, input_mask, trg = input_ids.cuda(), input_mask.cuda(), trg.cuda()
        output, _ = model.forward(input_ids, input_mask)
        loss = criterion(output, trg)

        loss.backward()
        opt.step()
        opt.zero_grad()
        
        total_loss += loss.item()
        cnt += 1
    return total_loss / cnt


def evaluate(dataloader, model, criterion, epoch=None, use_cuda=True):
    """Standard evaluation function.

    Executing an evaluation.

    Args:
        dataloader: pytorch DataLoader for handling evaluation.
        model: nn.Module to evaluate.
        criterion: nn.criterion for the loss function of evaluation.
        epoch: indicate the specific epoch at this training.
        use_cuda: whether use cuda.

    Returns:
        Evaluation results including average loss and average ACC.

        total_loss / cnt, total_acc / cnt
    """
    model.eval()
    total_loss = 0
    total_acc = 0
    cnt = 0
    desc = "Evaluating (Epoch {:d})".format(epoch) if epoch is not None else "Evaluating"
    for i, batch_items in tqdm(enumerate(dataloader), total=dataloader.__len__(), leave=False, desc=desc):
        input_ids = batch_items['input_ids']
        input_mask = batch_items['input_mask']
        trg = batch_items['targets']
        if use_cuda:
            input_ids, input_mask, trg = input_ids.cuda(), input_mask.cuda(), trg.cuda()
        output, _ = model.forward(input_ids, input_mask)
        
        acc = (output.max(axis=1).indices == trg).sum().float() / len(trg)
        loss = criterion(output, trg)
        
        total_loss += loss.item()
        total_acc += acc.item()
        cnt += 1
    return total_loss / cnt, total_acc / cnt


def evalpred(dataloader, model, epoch=None, multi_label=True, use_cuda=True):
    """merged-Batch prediction.

    Return the predicted label and true_label for evaluation.

    Args:
        dataloader: pytorch DataLoader for handling evaluation,
        model: nn.Module to evaluate,
        epoch: indicate the specific epoch at this training,
        multi_label: whether to evaluate the multi label prediction,
        use_cuda: whether use cuda.

    Returns:
        numpy arrays for: 
            (1)predicted probability,
            (2)predicted label, (None) if multi_label is True,
            (3)true label 
        of all data.

        all_prob, all_pred, all_trg
    """
    model.eval()
    all_prob = []
    all_pred = []
    all_trg = []
    desc = "Evaluating (Epoch {:d})".format(epoch) if epoch is not None else "Evaluating"
    for i, batch_items in tqdm(enumerate(dataloader), total=dataloader.__len__(), leave=False, desc=desc):
        input_ids = batch_items['input_ids']
        input_mask = batch_items['input_mask']
        trg = batch_items['targets']
        if use_cuda:
            input_ids, input_mask, trg = input_ids.cuda(), input_mask.cuda(), trg.cuda()
        output, _ = model.forward(input_ids, input_mask)
        all_prob.append(output.cpu().detach())
        all_pred.append(output.max(axis=1).indices) if not multi_label else None
        all_trg.append(trg)
    if multi_label:
        all_prob = torch.sigmoid(torch.cat(all_prob)).numpy()
    else:
        all_prob = F.softmax(torch.cat(all_prob), dim=1)[:, 1].numpy()
    all_pred = torch.cat(all_pred).cpu().numpy() if not multi_label else None
    all_trg = torch.cat(all_trg).cpu().numpy()
    model.train()
    return all_prob, all_pred, all_trg


def getperf(y_prob, y_pred, y_true):
    """retrieve performance results.

    get performance results from the prediction.

    Args:
        y_prob: predicted probabilities, ndarray.
        y_pred: predicted labels, ndarray.
        y_true: true labels, ndarray.

    Returns:
        A dict combining different performance metrics.
        {
            "CM": confusion matrix,
            'ACC': standard accuracy,
            'SEN': sensitivity(true positive rate),
            'PREC': precision,
            "SPEC": specificity (1 - false positive rate),
            "MCC": Matthews Correlation Coefficient,
            "F1": f1 score,
            "ROCCURVE": receiver operating curve,
            "AUCROC": aera under the ROC. 
        }

        perftab -> dict
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    aucroc = auc(fpr, tpr)
    perftab = {
        "CM": confusion_matrix(y_true, y_pred),
        'ACC': (tp + tn) / (tp + fp + fn + tn),
        'SEN': tp / (tp + fn),
        'PREC': tp / (tp + fp),
        "SPEC": tn / (tn + fp),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROCCURVE": {'fpr':fpr, 'tpr':tpr},
        "AUCROC": aucroc
    }
    return perftab


def perf_multi_label(y_prob, y_true, label_names:list=[], thresholds:Union[float, list]=0.5):
    """retrieve multi-label performance results.

    get performance results from the prediction under multi-label prediction.

    Args:
        y_prob: predicted probabilities, ndarray.
        y_true: true labels, ndarray.
        label_names: String list that indicates the names of labels.
        thresholds: Union[float, list], probability thresholds for predicted labels. 

    Returns:
        rslttab -> {[label_name: 
            {
                'SEN': sensitivity(true positive rate),
                "SPEC": specificity (1 - false positive rate),
                "GMean": sqrt(SEN * SPEC),
                "AUCROC": aera under the ROC. 
            }
        ]}
        rsltcms -> {[label_name: {'tn':tn, 'fp':fp, 'fn':fn, 'tp':tp}]}
        rsltrocs -> {[label_name: {'fpr':fpr, 'tpr':tpr, 'thresholds':throc}]}
    """
    assert y_prob.shape == y_true.shape
    num_labels = y_true.shape[1]
    if len(label_names) != num_labels:
        warn("label_names (list) is not assigned in proper, using default setting.")
        label_names = ["label_{:d}".format(ll) for ll in range(num_labels)]
    if isinstance(thresholds, list) and (len(thresholds) != num_labels):
        warn("Incorrect length of thresholds. Using 0.5 instead.")
        thresholds=0.5
    rslttab = dict()
    rsltrocs = dict()
    rsltcms = dict()
    for ii in range(num_labels):
        l_prob = y_prob[:, ii]
        l_true = y_true[:, ii]
        if isinstance(thresholds, list):
            l_pred = (l_prob > thresholds[ii]).astype(int)
        else:
            l_pred = (l_prob > thresholds).astype(int)
            
        tn, fp, fn, tp = confusion_matrix(l_true, l_pred).ravel()
        sen = tp / (tp + fn)
        spec = tn / (tn + fp)
        gmean = math.sqrt(sen * spec)
        fpr, tpr, throc = roc_curve(l_true, l_prob)
        aucroc = auc(fpr, tpr)

        rslttab[label_names[ii]] = {
            "SEN": sen,
            "SPEC": spec,
            "GMean": gmean,
            "AUCROC":aucroc
        }
        rsltrocs[label_names[ii]] = {'fpr':fpr, 'tpr':tpr, 'thresholds':throc}
        rsltcms[label_names[ii]] = {'tn':tn, 'fp':fp, 'fn':fn, 'tp':tp}
    rslttab = pd.DataFrame(rslttab)
    rsltcms = pd.DataFrame(rsltcms)
    return rslttab, rsltcms, rsltrocs


if __name__ == "__main__":
    print(torch.__version__)
    ## Create parser and construct args
    hstr = """%prog Training phase for multi-label antimicrobial peptide prediction."""
    parser = OptionParser(hstr, description='Training phase for multi-label antimicrobial peptide prediction.')

    parser.add_option('--cuda', action='store_true', dest='cuda', default=False, help='whether to use cuda')
    parser.add_option('--data-parallel', action='store_true', default=False, dest='data_parallel', help='whether to parallel the data on multiple-GPU')
    parser.add_option('--seed', action='store', type=int, default=810, dest='seed', help='random seed severed for training phase')
    parser.add_option('--shuffle', action='store_true', default=False, dest='shuffle', help='whether to shuffle training data')
    parser.add_option('--lr', action='store', type=float, default=0.04, dest='lr', help='initial learning rate of training')
    parser.add_option('--ckpt-iter', action='store', type=int, default=50, dest='ckpt_iter', help='iteration of saving checkpoints')
    parser.add_option('--task', action='store', type=str, default="AMP", dest='task')
    parser.add_option('-e', '--epochs', action='store', type=int, default=256, dest='epochs', help='epochs for training')
    parser.add_option('-b', '--batch-size', action='store', type=int, default=32, dest='batch_size', help='batch size for training')
    parser.add_option('-d', '--rslt-dir', action='store', type=str, default=None, dest='rslt_dir', help='directory name under the ./result/')
    
    (options,args)=parser.parse_args()

    USE_CUDA = options.cuda
    DATA_PARALLEL = options.data_parallel
    SHUFFLE = options.shuffle
    RANDOM_SEED = options.seed
    LR = options.lr
    CKPT_PER_ITER = options.ckpt_iter
    EPOCHS = options.epochs
    BS = options.batch_size
    RSLT_DIR = options.rslt_dir
    ASL_CONFIG_MULTIL = {'gamma_neg':2, 'gamma_pos':2}
    
    # Lantian Yao Revised
    MULTI_LABEL = False if options.task == "AMP" else True
    print(MULTI_LABEL)
    ###

    MODEL_PRESETS = {'linsize': 640, 'pretrained': True, 'bert_frozen': True, 'lindropout': 0.2}

    # Lantian Yao Revised
    if options.task == "AMP":
        LABELS = "AMP"
    else:
        LABELS = ["Anticancer", "Antifungal", "AntiGramn", "AntiGramp", "Antimammal", "Antiparasite","Antiviral"]
    ###

    if USE_CUDA:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ## Manipulate directory, write
    os.makedirs("results") if not os.path.exists("results") else None

    time_now = int(round(time.time() * 1000))
    time_now = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time_now / 1000))
    if RSLT_DIR is None:
        RSLT_DIR = "rslt_{}".format(time_now)
    try:
        cls_dir = os.path.join("results", RSLT_DIR)
        os.makedirs(cls_dir)
    except FileExistsError:
        warn("Result directory already exsits. create default directory.")
        cls_dir = os.path.join("results", "rslt_{}".format(time_now))
        os.makedirs(cls_dir)

    with open(os.path.join(cls_dir, "loss_parameters.txt"), 'w') as fw:
        fw.write(str(ASL_CONFIG_MULTIL))

    with open(os.path.join(cls_dir, "arguments.txt"), 'w+') as f:
        f.write("labels: {:s}\n".format(str(LABELS)))
        f.write("shuffle training data: {:s}".format(str(SHUFFLE)))
        f.write("epoches: {:d}\n".format(EPOCHS))
        f.write("learning_rate: {:.6f}\n".format(LR))
        f.write("use_cuda: {:s}\n".format(str(USE_CUDA)))
        f.write("random_seed:{:d}\n".format(RANDOM_SEED))
        f.write("model_presets:\n")
        f.write("{:s}".format(str(MODEL_PRESETS)))

    ## Load training data
    print("Loading training data..", flush=True, end=" ")
    # traindata = pd.read_csv("data/train/{:s}".format("stage-1.csv" if LABELS == "AMP" else "mtl.csv"))
    # trainset = AMPDataset(traindata, task_label=LABELS)
    if options.task == "AMP":
        traindata = pd.read_csv("data/train/stage-1.csv")
    else:
        traindata = pd.read_csv("data/train/{:s}".format("mtl.csv"))
    if SHUFFLE:
        traindata = shuffle(traindata, random_state=3500)
    trainset = AMPDataset(traindata, task_label=LABELS)
    print("Complete!", flush=True)

    ## Initialize model, optimizer, criterion and trainloader
    print("Initialize model training settlements", flush=True, end=" ")

    # LTY revised
    if MULTI_LABEL:
        model = BERTAMP(**MODEL_PRESETS, num_labels=len(LABELS))
    else:
        model = BERTAMP(**MODEL_PRESETS)
    ###
    if USE_CUDA:
        model = model.cuda()
        torch.cuda.manual_seed(RANDOM_SEED)
        if DATA_PARALLEL:
            device_ids = [0]
            model = nn.DataParallel(model, device_ids=device_ids)
    else:
        torch.manual_seed(RANDOM_SEED)
    opt = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.99), eps=1e-8)
    # sche = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.94)
    sche = torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[20, 40, 80, 90, 95, 120, 150, 170, 218],gamma = 0.75)

    #Lantian Yao Revise
    if MULTI_LABEL: 
        criterion = AsymmetricLossOptimized(**ASL_CONFIG_MULTIL)
    else:
        criterion = ASLSingleLabel(**ASL_CONFIG_MULTIL)
    ###

    trainloader = DataLoader(trainset,
                        batch_size=BS,
                        num_workers=16,
                        pin_memory=True,
                        collate_fn=trainset.collate_fn)

    trainlosses = []
    trainacces = []
    print("Complete!", flush=True)
    print()
    print("==============Training Session==============", flush=True)
    ## Proceed traning process
    ckpt_dir = os.path.join(cls_dir, "checkpoints")
    os.makedirs(ckpt_dir) if not os.path.exists(ckpt_dir) else None

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train(trainloader, model, criterion, opt, epoch=epoch, use_cuda=USE_CUDA)
        trainlosses.append(tr_loss)
        
        #Lantian Yao Revised
        y_prob, y_pred, y_true = evalpred(trainloader, model, epoch=epoch, use_cuda=USE_CUDA, multi_label=MULTI_LABEL)
        if MULTI_LABEL:
            y_pred = (y_prob > 0.5).astype(int)
            tr_acc = hamming_loss(y_true, y_pred)
        else:
            tr_acc = (y_pred == y_true).sum() / len(y_pred)
        ###

        trainacces.append(tr_acc)
        sche.step()  # step the learning_rate scheduler
        # record training process
        #Lantian Yao Revised
        print("Epoch {:d}: Loss: {:.2f}; {:s}: {:.2f}%, lr={:.2e}". \
            format(epoch,
                   tr_loss,
                   "Hamming Distance" if MULTI_LABEL else "Accuracy",
                   tr_acc * 100,
                   opt.state_dict()['param_groups'][0]['lr']))
        ###
        # save checkpoints
        if epoch % CKPT_PER_ITER == 0:
            print("Saving checkpoint for epoch {:d}.".format(epoch), flush=True)
            checkpoint_dict = {
            'model_presets': MODEL_PRESETS,
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if DATA_PARALLEL else model.state_dict(),
            'opt_state_dict': opt.state_dict(),
            'losses': trainlosses,
            'acces': trainacces,
            }
            #Lantian Yao Revise
            if MULTI_LABEL:
                mtl_ckpt_tab, _, mtl_ckpt_rocs = perf_multi_label(y_prob, y_true, label_names=LABELS)
                checkpoint_dict['mtl_temp_perf'] = mtl_ckpt_tab
                checkpoint_dict['mtl_temp_rocs'] = mtl_ckpt_rocs
                print("Classification Performance for epoch {:d}:".format(epoch))
                print(mtl_ckpt_tab)
            else:
                print(getperf(y_prob, y_pred, y_true))
            ###
            torch.save(checkpoint_dict, os.path.join(ckpt_dir, "epoch_{:d}.ckpt".format(epoch)))
            print("Saving Complete!", flush=True)


    print("==============Training Complete!============", flush=True)
    print("", flush=True)
    ## Evaluate training results
    print("=============Evaluate Training...===========", flush=True)
    #Lantian Yao Revise
    train_prob, train_pred, train_true = evalpred(trainloader, model, use_cuda=USE_CUDA, multi_label=MULTI_LABEL)  # Training results
    ###
    print("=============Evaluation Complete!===========", flush=True)
    final_dict = {
        'model_presets': MODEL_PRESETS,
        'model_state_dict': model.module.state_dict() if DATA_PARALLEL else model.state_dict(),  # converted to un-distributed model
        'train_losses': np.asarray(trainlosses),
        'train_accuracies': np.asarray(trainacces),
    }

    #Lantian Yao Revise
    if MULTI_LABEL:
        mtl_tab, mtl_cms, mtl_rocs = perf_multi_label(train_prob, train_true, label_names=LABELS)
        final_dict['mtl_perf'] = mtl_tab
        final_dict['mtl_confusion_matrix'] = mtl_cms
        final_dict['mtl_rocs'] = mtl_rocs
    else:
        final_dict['train_performance'] = getperf(train_prob, train_pred, train_true)
    ###
    torch.save(final_dict, os.path.join(cls_dir, "final_models_evals.pkl"))
