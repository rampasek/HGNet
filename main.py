#!/usr/bin/env python3
import json
import os
import pprint as pp
import random
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from torch_geometric.data import DataLoader

from dataset import load_dataset, split_dataset
from models.ops import train_epoch, evaluate, \
    train_transductive, evaluate_transductive, select_by_perf, checkpoint
from options import get_options


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(opts):
    print("[*] Options")
    pp.pprint(vars(opts))
    print("")

    # Set the random seed and improve reproducibility of CUDA functions' results
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # Optionally configure TensorBoard writer
    tb_writer = None
    if not opts.no_tensorboard:
        tb_writer = SummaryWriter(os.path.join(opts.log_dir, opts.dataset, opts.run_name))

    # Set the PyTorch device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Load data
    dataset, predef_splits, task = load_dataset(opts.dataset, opts)
    opts.task = task
    if task.is_inductive:
        if task.has_predefined_split:
            assert opts.fold == -1, "Cross-validation not available."
            test_dataset = predef_splits['test']
            val_dataset = predef_splits['val']
            train_dataset = predef_splits['train']
            dataset = train_dataset
        elif opts.fold == -1:  # random split
            dataset = dataset.shuffle()
            n = (len(dataset) + 9) // 10
            test_dataset = dataset[:n]
            val_dataset = dataset[n:2 * n]
            train_dataset = dataset[2 * n:]
        else:
            test_dataset, val_dataset, train_dataset = split_dataset(dataset, opts.fold)

        # Loaders
        train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=opts.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False)

        opts.epoch_size = len(train_dataset)
    else:
        assert len(dataset) == 1, \
            "Exactly one graph is expected in the transductive scenario."
        assert opts.dataset.startswith('WikiCS') or opts.fold == -1, \
            "Cross-validation not available in this transductive scenario."
        opts.epoch_size = len(dataset)

    # Use training loss based on dataset prediction task type
    if task.is_classification:
        if task.is_multi_task:
            # multi-task binary classification
            opts.loss_func = torch.nn.BCEWithLogitsLoss()
        else:
            # single-task multi-class classification
            opts.loss_func = torch.nn.CrossEntropyLoss()
    else:
        opts.loss_func = torch.nn.MSELoss()

    # Model selection criterion
    if task.evaluator is not None:
        # use the OGB Evaluators' metric if available
        measure = task.evaluator.eval_metric
        greater = (measure != 'rmse')
    else:
        if task.is_classification:
            measure, greater = ('accuracy', True)
        else:
            measure, greater = ('RMSE', False)

    # Initialize model
    print(f"[*] The model:")
    model = opts.model_class(dataset, task, opts).to(opts.device)
    print(model)
    num_model_params = count_parameters(model)
    print(f"total number of trainable params: {num_model_params}")
    print(f"task setting: {task}")

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.wd)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: opts.lr_decay ** epoch
    )

    # Start the training loop
    best_stats = {}
    best_ckpt = None
    if not opts.eval_only:
        print("\n[*] Training...")
        train_start_time = time.time()
        for epoch in range(opts.n_epochs):
            found_better = False
            if task.is_inductive:
                # Dataset contains multiple graphs i.e. inductive scenario

                if epoch == 0:
                    # evaluate(model, train_loader, epoch, tb_writer, opts, 'train')
                    evaluate(model, val_loader, epoch, tb_writer, opts, 'val')
                    evaluate(model, test_loader, epoch, tb_writer, opts, 'test')

                train_epoch(
                    model,
                    optimizer,
                    lr_scheduler,
                    epoch,
                    train_loader,
                    tb_writer,
                    opts
                )

                pstats = {'epoch': epoch, 'num_params': num_model_params}
                # pstats.update(evaluate(model, train_loader, epoch, tb_writer, opts, 'train'))
                pstats.update(evaluate(model, val_loader, epoch, tb_writer, opts, 'val'))
                pstats.update(evaluate(model, test_loader, epoch, tb_writer, opts, 'test'))
                best_stats, found_better = select_by_perf(
                    best_stats, pstats, 'val', measure, greater)

            else:
                # Dataset is a single (large) graph i.e. transductive scenario

                if epoch == 0:
                    evaluate_transductive(model, dataset[0], epoch, tb_writer, opts)

                train_transductive(
                    model,
                    optimizer,
                    lr_scheduler,
                    epoch,
                    dataset[0],
                    tb_writer,
                    opts
                )

                pstats = {'epoch': epoch, 'num_params': num_model_params}
                pstats.update(evaluate_transductive(model, dataset[0], epoch, tb_writer, opts))
                stopping_split = 'val' if not hasattr(dataset[0], 'stopping_mask') else 'stopping'  # WikiCS has a dedicated stopping mask
                best_stats, found_better = select_by_perf(
                    best_stats, pstats, stopping_split, measure, greater)

            if found_better:
                print(f"[ ] Found new best model")
                best_ckpt = checkpoint(model, optimizer, epoch, opts, to_variable=True)
                # Log the current best performance
                best_stats['epoch'] = epoch
                with open(os.path.join(opts.save_dir, "results-tmp.json"), 'w') as f:
                    json.dump(best_stats, f, indent=True)
            if opts.time_limit and time.time() - train_start_time > opts.time_limit * 3600:
                elapsed = time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start_time))
                print(f"[ ] Time limit exceeded after epoch {epoch} when {elapsed} elapsed.")
                break


    if opts.eval_only:
        # Load saved parameters
        assert opts.load_path is not None, "Must provide load_path to a trained model"
        if opts.load_path.endswith(".pt"):
            save_fname = opts.load_path
        else:
            save_fname = os.path.join(opts.load_path, "best.pt")
        print(f"\n[*] Loading model from {save_fname}")
        # Load all the saved tensors as CPU tensors first
        ckpt = torch.load(save_fname, map_location=lambda storage, loc: storage)
    else:
        # Save the selected trained model and prepare for its reloading
        torch.save(best_ckpt, os.path.join(opts.save_dir, "best.pt"))
        ckpt = best_ckpt

    # Overwrite the model parameters by parameters from a checkpoint
    # model.load_state_dict({**model.state_dict(), **ckpt.get('model_state_dict', {})})
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(opts.device)

    # Evaluate the final model on all datasets
    print(f"[*] Evaluating the model ...")
    epoch = 0 if opts.eval_only else best_stats['epoch']
    result_stats = {'epoch': epoch, 'num_params': num_model_params}
    if task.is_inductive:
        result_stats.update(evaluate(model, train_loader, epoch, tb_writer, opts, 'train'))
        result_stats.update(evaluate(model, val_loader, epoch, tb_writer, opts, 'val'))
        result_stats.update(evaluate(model, test_loader, epoch, tb_writer, opts, 'test'))
    else:
        result_stats.update(evaluate_transductive(model, dataset[0], epoch, tb_writer, opts))

    if not opts.eval_only:
        # sanity check
        pp.pprint(result_stats)
        pp.pprint(best_stats)

    # Log hyper-parameters and performance
    with open(os.path.join(opts.save_dir, "results.json"), 'w') as f:
        json.dump(result_stats, f, indent=True)
    if not opts.no_tensorboard:
        hparam_dict = {
            'model': opts.model, 'dataset': opts.dataset,
            'lr': opts.lr, 'wd': opts.wd,
            # 'batch_size': opts.batch_size,
            'n_epochs': opts.n_epochs, 'best_epoch': result_stats['epoch']
        }
        if task.is_classification:
            metric_dict = {
                'hparam/accuracy': result_stats['test/accuracy'],
                'hparam/AUROC': result_stats['test/AUROC'],
                'hparam/loss': result_stats['test/loss']
            }
        else:
            metric_dict = {
                'hparam/RMSE': result_stats['test/RMSE'],
                'hparam/MAE': result_stats['test/MAE'],
                'hparam/loss': result_stats['test/loss']
            }
        # Write HParams manually into the opened TB log instead of calling
        # its add_hparams method that creates a new TB log file
        exp, ssi, sei = hparams(hparam_dict, metric_dict)
        tb_writer._get_file_writer().add_summary(exp)
        tb_writer._get_file_writer().add_summary(ssi)
        tb_writer._get_file_writer().add_summary(sei)
        for k, v in metric_dict.items():
            tb_writer.add_scalar(k, v)
        tb_writer.flush()

if __name__ == "__main__":
    start_time = time.time()
    main(get_options())
    duration = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
    print(f"Total elapsed wall-clock time: {duration}s")
