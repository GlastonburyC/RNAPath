import numpy as np
import torch
from utils.utils import *
import os
from datasets.dataset_generic import save_splits
from HE2RNA_GAMIL_all_genes.models.model_RNAPath import RNAPath
import pandas as pd
import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
import scipy
from sklearn.metrics import mean_squared_error as mse
import random



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=20, verbose=False):
        
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.r_score_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss 

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Val loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss



def train(datasets, cur, args):
    """   
        training main function
    """
    print('\nRNAPath training started!')
    
    # Path to the results dir where the results will be stored (e.g. /results/exp_code/i/)
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)
    
    # Tensorboard logging
    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit MSE loss function...', end=' ')
    # Loss function initialization
    loss_fn = nn.MSELoss()
    if device.type == 'cuda':
        loss_fn = loss_fn.cuda()
    print('Done!', flush=True)
    
    print('\nInit Model...', end=' ')
    model_dict = {'n_classes': args.n_classes}


    # Model creation
    
    model = RNAPath(**model_dict)

    # Move the model to device
    model.to(device)
    print('Model ready!', flush=True)


    print('\nInit optimizer and lr scheduler ...', end=' ')
    
    # Optimizer
    optimizer = get_optim(model, args)
    # Scaler
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    # Scheduler
    if args.lr_scheduler != 'constant':
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    print('Done!', flush=True)

    
    print('\nInit Loaders...', end=' ')
    
    # Data loaders
    train_loader = get_split_loader(train_split, training=True)
    val_loader = get_split_loader(val_split)
    test_loader = get_split_loader(test_split)
    print('Done!', flush=True)

    print('\nSetup EarlyStopping...', end=' ')
    # Early Stopping
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 20, stop_epoch=30, verbose = True)


        
    print('Done!', flush=True)

    print(f'Number of genes for {args.tissue_code} : {args.n_classes}')

    lis = list(range(args.n_classes))

    # Batch size of genes; we can't train all the linear regressors in a single iteration due to memory limitations, 
    # therefore we split the genes in batches
    batch_size = 200
    batches = [lis[i:i+batch_size] for i in range(0, len(lis), batch_size)]

    print(f'Number of gene batches: {len(batches)}')

    # Model Training
    for epoch in range(args.max_epochs):

        random.shuffle(lis)
        # create batches of genes (due to memory limitations, we cannot train all the regressors
        # We have #genes/batch_size iterations and accumulate gradients to just perform a single update step per sample)
        batches = [lis[i:i+batch_size] for i in range(0, len(lis), batch_size)]
        start = datetime.datetime.now()
        train_loop(epoch, model, train_loader, optimizer, scaler, args.n_classes, writer, loss_fn, batches)
        end = datetime.datetime.now()
        print(f'Time for training epoch {epoch} : {end - start}', flush = True)
        stop, val_loss = validate(cur, epoch, model, val_loader, args.n_classes, early_stopping, writer, loss_fn, args.results_dir, tissue_code=args.tissue_code)
        
        if args.lr_scheduler != 'constant':
            scheduler.step(val_loss)
        
        if stop:
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    val_median_r, val_error, = summary(model, val_loader, args.n_classes, args.results_dir, args.tissue_code, split = 'val')
    print('Val error: {:.4f}, median r score: {:.4f}'.format(val_error, val_median_r))

    test_median_r, test_error = summary(model, test_loader, args.n_classes, args.results_dir, args.tissue_code, split='test')
    print('Test error: {:.4f}, median r score: {:.4f}'.format(test_error, test_median_r))

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.close()
    
    return val_median_r, val_error, test_median_r, test_error


def train_loop(epoch, model, loader, optimizer, scaler, n_classes, writer = None, loss_fn = None, gene_batches = None):
    # setting device   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    # Accuracy Logger
    train_loss = 0.
    torch.cuda.empty_cache()
    print('\n')


    for batch_idx, (data, label) in enumerate(loader):
        batch_loss = 0.

        # move to device
        data, label = data.to(device), label.to(device)
        
        optimizer.zero_grad()


        # HER2NA training
        # with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
        #     logits, ks = model(data.unsqueeze(0), training=True)
        #     logits = logits.squeeze(0)
        #     loss = loss_fn(logits, label.squeeze(0))
        # loss_value = loss.item()
        # train_loss += loss_value
        
        # scaler.scale(loss).backward()
        # batch_loss = loss_value

        #for each batch of genes
        for i in range(int(len(gene_batches))):

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):

                # RNA-Seq regression for genes in gene_batches(i)
                logits = model(data, genes_list = gene_batches[i])
                # compute loss
                loss = loss_fn(logits, label.squeeze(0)[gene_batches[i]])
            
            # divide loss by number of gene batches for gradient accumulation
            batch_loss += (loss.item()/len(gene_batches))
            
            loss_value = loss.item()
            train_loss += loss_value

            # gradient accumulation
            scaler.scale(loss).backward()
        
        #step
        scaler.step(optimizer)
        scaler.update()
            
        if (batch_idx) % 20 == 0:
            print('Training - batch {}, loss: {:.4f}, bag_size: {}, lr:{:.6f}:'.format(batch_idx, batch_loss, data.size(0), optimizer.param_groups[0]['lr']), flush = True) #loss_value
        
    # print loss for current epoch
    train_loss /= len(loader)
    print('Epoch: {}, train_loss: {:.4f}'.format(epoch, train_loss))

    
    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)

   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None, tissue_code=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss = 0.
    val_error = 0.

    genes_df = pd.read_csv(f'./resources/gene_set_{tissue_code}.txt', sep=' ')
    genes_list = genes_df['gene_id'].tolist()
    genes_descriptions = genes_df['gene_desc'].tolist()

    with torch.no_grad():
        pred = []
        gt = []
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
            logits = model(data.unsqueeze(0))
            gt.append(label[0].cpu().numpy())
            pred.append(logits.cpu().numpy()) 
            loss = loss_fn(logits, label.squeeze(0)) 
            val_loss += loss.item()
            if (batch_idx + 1) % 20 == 0:
                print('Validation - batch {}, loss: {:.4f}, bag_size: {}'.format(batch_idx, loss.item(), data.size(0)), flush = True)
            error = torch.nn.MSELoss()(logits, label.squeeze(0))
            val_error += error            
        
    val_error /= len(loader)
    val_loss /= len(loader)

    best = early_stopping.best_score

    if best == None or -val_loss >= best:

        rscores = []
        
        mode = 'w'

        with open(os.path.join(results_dir, 'report.txt'), mode) as f:
            f.write(f'gene_desc gene_id r_score mse p_value')
            f.write('\n')
            mode = 'a'

            
        for k in range(n_classes):
            p = np.array(pred)[:, k]
            g = np.array(gt)[:, k]
            
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(p, g)
            with open(os.path.join(results_dir, 'report.txt'), mode) as f:
                    try:
                        f.write(f'{genes_descriptions[k]} {genes_list[k]} {r_value} {mse(g,p)} {p_value}')
                        f.write('\n')
                    except ValueError:
                        pass

            rscores.append(r_value)

        print(f"Epoch {epoch} -->  median r score: {np.median(np.array(rscores))}")
                
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)

    print('\nVal Set, val_loss: {:.4f}'.format(val_loss))


    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True, val_loss

    return False, val_loss


def summary(model, loader, n_classes, results_dir=None, tissue_code=None, split=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    loss_fn = nn.MSELoss()
    if device.type == 'cuda':
        loss_fn = loss_fn.cuda()

    model.eval()

    mode = 'w'

    test_loss = 0.
    test_error = 0.

    rscores = []

    genes_df = pd.read_csv(f'./resources/gene_set_{tissue_code}.txt', sep=' ')
    genes_list = genes_df['gene_id'].tolist()
    genes_descriptions = genes_df['gene_desc'].tolist()
    
    
    with torch.no_grad():
        pred = []
        gt = []
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
            logits, _ = model(data.unsqueeze(0))
            logits = logits.squeeze(0)
            gt.append(label[0].cpu().numpy())
            pred.append(logits.cpu().numpy()) 
            loss = loss_fn(logits, label.squeeze(0)) 
            test_loss += loss.item()
            if (batch_idx + 1) % 20 == 0:
                print('Validation - batch {}, loss: {:.4f}, bag_size: {}'.format(batch_idx, loss.item(), data.size(0)), flush = True)
            error = torch.nn.MSELoss()(logits, label.squeeze(0))
            test_error += error            
        
    test_error /= len(loader)
    test_loss /= len(loader)
    
    with open(os.path.join(results_dir, f'report_{split}.txt'), mode) as f:
        f.write(f'gene_desc gene_id r_score mse p_value')
        f.write('\n')
        mode = 'a'
    
    data = []
    rscores = []

    for k in range(n_classes):
        p = np.array(pred)[:, k]
        g = np.array(gt)[:, k]
            
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(p, g)
        with open(os.path.join(results_dir, f'report_{split}.txt'), mode) as f:
            f.write(f'{genes_descriptions[k]} {genes_list[k]} {r_value} {mse(g,p)} {p_value}')
            f.write('\n')

        rscores.append(r_value)

    median_r_score = np.median(np.array(rscores))



    return median_r_score, test_error
