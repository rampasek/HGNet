import copy
import os
import time

import numpy as np
import torch
from sklearn import metrics
from tqdm import tqdm


def train_epoch(
        model,
        optimizer,
        lr_scheduler,
        epoch,
        train_loader,
        tb_writer,
        opts
):
    print(f"Start train epoch {epoch}, lr={optimizer.param_groups[0]['lr']} for run {opts.run_name}")
    step = epoch * ((opts.epoch_size + opts.batch_size - 1) // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_writer.add_scalar('learn-rate', optimizer.param_groups[0]['lr'], step)

    # Training for an epoch
    model.train()
    loss_epoch_all = 0
    for batch_ind, data in enumerate(tqdm(train_loader, disable=opts.no_progress_bar)):
        loss_epoch_all += train_batch(
            model,
            optimizer,
            epoch,
            batch_ind,
            step,
            data,
            tb_writer,
            opts
        )
        step += 1

    # lr_scheduler should be called at end of an epoch
    lr_scheduler.step()

    checkpoint(model, optimizer, epoch, opts)

    epoch_duration = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
    print(f"Finished in {epoch_duration}s", end='\t')
    print(f"Average epoch loss: {loss_epoch_all / len(train_loader.dataset):.4f}")


def train_batch(
        model,
        optimizer,
        epoch,
        batch_ind,
        step,
        data,
        tb_writer,
        opts
):
    data = data.to(opts.device)

    # Run forward pass through the model
    optimizer.zero_grad()
    pred = model(data)

    # Calculate the loss (and ignore nan targets i.e. unlabeled)
    is_labeled = data.y == data.y
    loss = opts.loss_func(pred[is_labeled], data.y[is_labeled])
    loss_all = data.y.size(0) * loss.item()
    # print(f"step: {step}  train/loss: {loss.item():.4f}")
    if tb_writer:
        tb_writer.add_scalar('batch-train/loss', loss, step)

    # Make backward pass and an optimization step
    loss.backward()
    optimizer.step()

    # Logging performance stats
    if opts.log_step != 0 and step % opts.log_step == 0:
        # print(f">> epoch: {epoch}, train_batch_ind: {batch_ind}")
        # Monitor norms of gradients
        log_grad_norms(model, step, tb_writer)

        if opts.task.is_classification:
            # Classification performance stats
            classification_perf(data.y, pred.detach(), model.task,
                                step, tb_writer, 'batch-train', False)
        else:
            # Regression performance stats
            regression_perf(data.y, pred, step, tb_writer, 'batch-train', False)
    return loss_all


def evaluate(model, loader, step, tb_writer, opts, mode='train'):
    y_true = []
    y_pred = []
    loss_all = 0

    model.eval()
    with torch.no_grad():
        for data in loader:
            data = data.to(opts.device)
            pred = model(data)

            is_labeled = data.y == data.y
            loss = opts.loss_func(pred[is_labeled], data.y[is_labeled])
            loss_all += data.y.size(0) * loss.item()

            y_true.append(data.y.detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    avg_loss = loss_all / len(loader.dataset)

    print(f"epoch: {step}  {mode}/loss: {avg_loss:.4f}", end='\t')
    if tb_writer:
        tb_writer.add_scalar(f"{mode}/loss", avg_loss, step)

    if opts.task.is_classification:
        pstats = classification_perf(y_true, y_pred, model.task,
                                     step, tb_writer, mode)
    else:
        pstats = regression_perf(y_true, y_pred, step, tb_writer, mode)
    pstats[f'{mode}/loss'] = avg_loss

    return pstats


def train_transductive(
        model,
        optimizer,
        lr_scheduler,
        epoch,
        data,
        tb_writer,
        opts
):
    print(f"Start train epoch {epoch}, lr={optimizer.param_groups[0]['lr']:.5f}")
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_writer.add_scalar('learn-rate', optimizer.param_groups[0]['lr'], epoch)

    # Training for an epoch
    model.train()
    data = data.to(opts.device)
    loss = opts.loss_func(model(data)[data.train_mask], data.y[data.train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Monitor norms of gradients
    log_grad_norms(model, epoch, tb_writer)

    # lr_scheduler should be called at end of an epoch
    lr_scheduler.step()

    checkpoint(model, optimizer, epoch, opts)

    epoch_duration = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
    print(f"Finished in {epoch_duration}s with train/loss: {loss.item():.4f}")


def evaluate_transductive(model, data, step, tb_writer, opts):
    model.eval()
    data = data.to(opts.device)
    with torch.no_grad():
        pred = model(data)

    pstats = {}
    splits = ['train', 'val', 'test']
    if hasattr(data, 'stopping_mask'):
        splits.append('stopping')
    for split in splits:
        mask = getattr(data, split + '_mask')
        y = data.y[mask]

        loss = opts.loss_func(pred[mask], y.view(-1))
        pstats[f'{split}/loss'] = loss.item()
        # print(f"epoch: {step}  {split}/loss: {loss.item():.4f}")
        if tb_writer:
            tb_writer.add_scalar(f"{split}/loss", loss, step)

        if opts.task.is_classification:
            pstats.update(classification_perf(y, pred[mask].detach(), model.task,
                                              step, tb_writer, mode=split))
        else:
            pstats.update(regression_perf(y, pred[mask].detach(),
                                          step, tb_writer, mode=split))

    return pstats


def classification_perf(y_true, logit, task,
                        step, tb_writer, mode='train', verbose=True):
    y_true = y_true.detach().cpu().numpy()
    logit = logit.detach().cpu()
    if task.is_multi_task:
        proba = torch.sigmoid(logit).numpy()
        pred = (proba > 0.5).astype(int)
    else:
        proba = torch.nn.functional.softmax(logit, dim=-1).numpy()
        pred = np.argmax(proba, axis=-1)
    # print("   y_true:    ", y_true, y_true.sum())
    # print("   preds:", proba[:5], pred[:10], proba.sum(), pred.sum())

    ### accuracy
    acc_list = []
    if task.is_multi_task:
        for i in range(y_true.shape[1]):
            is_labeled = y_true[:, i] == y_true[:, i]
            acc_list.append(metrics.accuracy_score(y_true[is_labeled, i],
                                                   pred[is_labeled, i]))
    else:
        acc_list.append(metrics.accuracy_score(y_true, pred))
    accuracy = sum(acc_list) / len(acc_list)

    ### F1
    f1_list = []
    if task.is_multi_task:
        for i in range(y_true.shape[1]):
            # F1 is only defined when there is at least one positive
            if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
                # ignore nan values
                is_labeled = y_true[:, i] == y_true[:, i]
                f1_list.append(metrics.f1_score(y_true[is_labeled, i],
                                                pred[is_labeled, i]))
    else:
        try:
            f1_list.append(metrics.f1_score(y_true, pred, average='macro'))
        except:
            pass
    if len(f1_list) == 0:
        # No positively labeled data available. Cannot compute ROC-AUC.
        f1 = np.nan
    else:
        f1 = sum(f1_list) / len(f1_list)

    ### AUROC
    auroc_list = []
    if task.is_multi_task:
        for i in range(y_true.shape[1]):
            # AUC is only defined when there is at least one positive
            if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
                # ignore nan values
                is_labeled = y_true[:, i] == y_true[:, i]
                auroc_list.append(metrics.roc_auc_score(y_true[is_labeled, i],
                                                        proba[is_labeled, i]))
    else:
        try:
            if proba.shape[1] == 2:  # single binary classification
                auroc_list.append(metrics.roc_auc_score(y_true, proba[:, 1]))
            else:
                auroc_list.append(metrics.roc_auc_score(y_true, proba, multi_class='ovo'))
        except:
            pass
    if len(auroc_list) == 0:
        # No positively labeled data available. Cannot compute ROC-AUC.
        auroc = np.nan
    else:
        auroc = sum(auroc_list) / len(auroc_list)

    ### OGB Evaluator
    evaluator_stats = {}
    if task.evaluator is not None:
        input_dict = {"y_true": y_true, "y_pred": logit.numpy()}
        try:
            evaluator_stats = task.evaluator.eval(input_dict)
        except Exception as e:
            print(f"Ignoring an exception in Evaluator: {e}")
            pass

    if verbose:
        s = f"> {mode[:5]}\t accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUROC: {auroc:.4f}"
        for key, val in evaluator_stats.items():
            s += f" {key}: {val:.4f}"
        print(s)

    # Log values to TensorBoard
    if tb_writer:
        tb_writer.add_scalar(f'{mode}/accuracy', accuracy, step)
        tb_writer.add_scalar(f'{mode}/F1', f1, step)
        tb_writer.add_scalar(f'{mode}/AUROC', auroc, step)
        # tb_writer.add_scalar(f'{mode}/AUPR', aupr, step)
        for key, val in evaluator_stats.items():
            tb_writer.add_scalar(f'{mode}/{key}', val, step)

    all_stats = {f'{mode}/accuracy': accuracy,
                 f'{mode}/F1': f1,
                 f'{mode}/AUROC': auroc}
    for key, val in evaluator_stats.items():
        all_stats[f'{mode}/{key}'] = val
    return all_stats


def regression_perf(y_true, y_pred, step, tb_writer, mode='train', verbose=True):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    # print("   y:   ", y_true[:5], y_true.shape)
    # print("   pred:", y_pred[:5], y_pred.shape)

    r2_score = metrics.r2_score(y_true, y_pred)
    RMSE = metrics.mean_squared_error(y_true, y_pred, squared=False)
    MAE = metrics.mean_absolute_error(y_true, y_pred)

    if verbose:
        print(f"> {mode}\t R2: {r2_score:.4f}, RMSE: {RMSE:.4f}, MAE: {MAE:.4f}")
    if tb_writer:
        tb_writer.add_scalar(f'{mode}/R2', r2_score, step)
        tb_writer.add_scalar(f'{mode}/RMSE', RMSE, step)
        tb_writer.add_scalar(f'{mode}/MAE', MAE, step)

    return {
        f'{mode}/R2': r2_score,
        f'{mode}/RMSE': RMSE,
        f'{mode}/MAE': MAE
    }


def log_grad_norms(model, step, tb_writer):
    grad_norms = [p.grad.data.detach().norm(2).cpu().item()
                  for p in model.parameters() if p.grad is not None]
    avg_grad_norm = np.mean(grad_norms)
    # print(f"avg_grad_norm: {avg_grad_norm:.4f}")
    if tb_writer:
        tb_writer.add_scalar('avg_grad_norm', avg_grad_norm, step)

    # Report parameters without gradient
    # for p_name, p in model.named_parameters():
    #     if p.requires_grad and p.grad is None:
    #         print(f"No gradient for parameter: {p_name}")


def checkpoint(model, optimizer, epoch, opts, save_name=None, to_variable=False):
    if to_variable:
        return {
            'model_state_dict': copy.deepcopy(model.state_dict()),
            'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
            'rng_state': copy.deepcopy(torch.get_rng_state()),
            'cuda_rng_state': copy.deepcopy(torch.cuda.get_rng_state_all())
        }
    elif (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) \
            or epoch == opts.n_epochs - 1 or save_name is not None:
        print("Saving model and state...")
        file_name = f"epoch-{epoch}.pt" if save_name is None else f"{save_name}.pt"
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all()
            },
            os.path.join(opts.save_dir, file_name)
        )

def select_by_perf(current_stats, candidate_stats,
                   split='val', measure='accuracy', greater=True):
    """
    Based on validation set performance retain or update the current best seen
    performance statistics dictionary.

    :param current: current best performance stats
    :param candidate: new candidate performance stats
    :param split: name of the data split on follow performance on
    :param measure: name of the performance metric to track
    :param greater: if True select the candidate if its perf. is greater
    :return: [stats dictionary, True if candidate is better than current]
    """
    if f'{split}/{measure}' not in current_stats:
        return candidate_stats, True
    found = candidate_stats[f'{split}/{measure}'] > current_stats[f'{split}/{measure}']
    if not greater:
        found = not found
    if found:
        return candidate_stats, True
    else:
        return current_stats, False


# def predict(model, dataset, opts):
#     print('\n[*] Inference on unlabeled data...')
#
#     model.eval()
#     x = dataset[:][0]
#     x = x.to(opts.device)
#     # Run as one batch
#     _, _, class_proba = model(x)
#
#     return class_proba
