import argparse
import logging
import os
import sys
from datetime import datetime
import random
import re

import pandas as pd
import multiprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from accelerate import Accelerator
from accelerate.logging import get_logger
from data import WikiDataset
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from accelerate import DistributedDataParallelKwargs
from torch.nn import Sequential
import gc

multiprocess.set_start_method("spawn", force=True)

def freeze_model(model, architecture, freeze_layers):
    if freeze_layers == 0:
        return model
    if architecture == 'BERT':
        for param in model.base_model.embeddings.parameters():
            param.requires_grad = False
        for param in model.base_model.encoder.layer[:freeze_layers].parameters():
            param.requires_grad = False
    elif architecture == 'RoBERTa':
        for param in model.base_model.embeddings.parameters():
            param.requires_grad = False
        for param in model.base_model.encoder.layer[:freeze_layers].parameters():
            param.requires_grad = False
    elif architecture == 'T5':
        for param in model.embed_tokens.parameters():
            param.requires_grad = False
        for param in model.block[:freeze_layers].parameters():
            param.requires_grad = False
    return model

def unfreeze_model(model, architecture):
    if architecture == 'BERT':
        for param in model.base_model.embeddings.parameters():
            param.requires_grad = True
        for param in model.base_model.encoder.layer.parameters():
            param.requires_grad = True
    elif architecture == 'RoBERTa':
        for param in model.base_model.embeddings.parameters():
            param.requires_grad = True
        for param in model.base_model.encoder.layer.parameters():
            param.requires_grad = True
    elif architecture == 'T5':
        for param in model.embed_tokens.parameters():
            param.requires_grad = True
        for param in model.block.parameters():
            param.requires_grad = True
    return model
                


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_name', type=str,
                        default='bert-base-cased', help='Model name or path to model')
    parser.add_argument('--model_architecture', type=str, choices=['BERT', 'RoBERTa', 'T5'], default='BERT', help='Model architecture')
    parser.add_argument('--data_dir', type=str,
                        required=True, help='Data directory')
    parser.add_argument('--num_epochs', type=int,
                        default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of workers for dataloader')
    parser.add_argument('--lr', type=float,
                        default=1e-5, help='Learning rate')
    parser.add_argument('--gamma_lr', type=float,
                        default=0.9, help='Gamma for lr scheduler')
    parser.add_argument('--print_steps', type=int, default=1_000,
                        help='Number of steps between printing loss')
    parser.add_argument('--save_steps', type=int, default=5_000,
                        help='Number of steps between saving model')
    parser.add_argument('--eval_steps', type=int, default=5_000,
                        help='Number of steps between evaluating model on validation set')
    parser.add_argument('--scheduler_steps', type=int, default=10_000,
                        help='Number of steps between scheduler steps')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint (needs --checkpoint_dir)')
    parser.add_argument('--checkpoint_dir', type=str,
                        help='Directory with checkpoint to resume training from')
    parser.add_argument('--ga_steps', type=int, default=1,
                        help='Number of steps for gradient accumulation')
    parser.add_argument('--full_freeze_steps', type=int, default=0,
                        help='Number of steps to freeze all layers except classification head (and link fuser if use_current_links is set)')
    parser.add_argument('--freeze_layers', type=int,
                        default=2, help='Number of initial layers to freeze')
    parser.add_argument('--head_lr_factor', type=float,
                        default=1, help='Factor for learning rate of classification head (and link fuser if use_current_links is set)')
    parser.add_argument('--max_tokens', type=int, default=256,
                        help='Maximum number of tokens')
    parser.add_argument('--neg_samples_train', type=int, default=1,
                        help='Number of negative samples for training')
    parser.add_argument('--neg_samples_eval', type=int, default=20,
                        help='Number of negative samples for evaluation')
    parser.add_argument('--temperature', type=float, default=1,
                        help='Temperature for softmax')
    parser.add_argument('--insert_mentions', action='store_true',
                         help='Where to insert mention knowledge') ################################## Added insert mentions
    parser.add_argument('--insert_section', action='store_true',
                        help='Whether to insert section title')
    parser.add_argument('--split_models', action='store_true',
                        help='If set, use a different encoder for articles and contexts')
    parser.set_defaults(resume=False, insert_mentions=False, insert_section=False, split_models=False)

    args = parser.parse_args()

    # check if checkpoint_dir is provided and exists if resuming training
    if args.resume:
        if args.checkpoint_dir is None:
            raise ValueError(
                "Please provide checkpoint directory with --checkpoint_dir")
        if not os.path.exists(args.checkpoint_dir):
            raise ValueError(
                f"Checkpoint directory {args.checkpoint_dir} does not exist")

    # check if data_dir exists
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory {args.data_dir} does not exist")

    # initialize accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    # set-up tensorboard
    if not os.path.exists('runs'):
        os.makedirs('runs', exist_ok=True)
    # take date time down to milliseconds
    date_time = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
    tb_dir = os.path.join('runs', date_time)
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir, exist_ok=True)
    if accelerator.is_main_process:
        writer = SummaryWriter(tb_dir)

    # create directory for logs and checkpoints
    if not os.path.exists('output_list_softmax'):
        os.makedirs('output_list_softmax', exist_ok=True)

    output_dir = os.path.join('output_list_softmax', date_time)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # create logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(os.path.join(output_dir, 'log.txt')),
                                  logging.StreamHandler()])
    logger = get_logger(__name__, log_level="INFO")

    # log arguments
    logger.info("Arguments:")
    for arg in vars(args):
        logger.info(f"\t- {arg}: {getattr(args, arg)}")

    if args.resume:
        logger.info("Loading model")
        try:
            if args.split_models:
                model_articles = AutoModel.from_pretrained(
                    os.path.join(args.checkpoint_dir, 'model_articles'))
                model_contexts = AutoModel.from_pretrained(
                    os.path.join(args.checkpoint_dir, 'model_contexts'))
            else:
                model = AutoModel.from_pretrained(args.checkpoint_dir, 'model')
            logger.info("Model loaded from checkpoint directory")
        except OSError:
            logger.info("Could not load model from checkpoint directory")
            logger.info("Initializing model from provided model name")
            if args.split_models:
                model_articles = AutoModel.from_pretrained(args.model_name)
                model_contexts = AutoModel.from_pretrained(args.model_name)
            else:
                model = AutoModel.from_pretrained(args.model_name)
        if args.split_models:
            model_contexts_size = model_contexts.config.hidden_size
            model_articles_size = model_articles.config.hidden_size
        else:
            model_contexts_size = model.config.hidden_size
            model_articles_size = model.config.hidden_size
        try:
            classification_head = Sequential(nn.Linear(model_articles_size * 2 + model_contexts_size, model_articles_size),
                                             nn.ReLU(),
                                             nn.Linear(model_articles_size, 1))
            classification_head.load_state_dict(torch.load(os.path.join(
                args.checkpoint_dir, 'classification_head.pth'), map_location='cpu'))
            logger.info("Classification head loaded from checkpoint directory")
        except OSError:
            logger.info(
                "Could not load classification head from checkpoint directory")
            logger.info("Initializing classification head with random weights")
            classification_head = Sequential(nn.Linear(model_articles_size * 2 + model_contexts_size, model_articles_size),
                                             nn.ReLU(),
                                             nn.Linear(model_articles_size, 1))
    else:
        logger.info("Initializing model")
        if args.split_models:
            model_articles = AutoModel.from_pretrained(args.model_name)
            model_contexts = AutoModel.from_pretrained(args.model_name)
            model_articles_size = model_articles.config.hidden_size
            model_contexts_size = model_contexts.config.hidden_size
        else:
            model = AutoModel.from_pretrained(args.model_name)
            model_articles_size = model.config.hidden_size
            model_contexts_size = model.config.hidden_size

        classification_head = Sequential(nn.Linear(model_articles_size * 2 + model_contexts_size, model_articles_size),
                                         nn.ReLU(),
                                         nn.Linear(model_articles_size, 1))
    
    # Remove decoder layers if present
    if args.model_architecture == 'T5':
        if args.split_models:
            model_articles = model_articles.encoder
            model_contexts = model_contexts.encoder
        else:
            model = model.encoder

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.save_pretrained(os.path.join(output_dir, 'tokenizer'))

    logger.info("Initializing optimizer")
    if args.split_models:
        optimizer = optim.Adam(model_articles.parameters(), lr=args.lr)
        optimizer.add_param_group(
            {'params': model_contexts.parameters(), 'lr': args.lr})
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # add classification head to optimizer
    optimizer.add_param_group(
        {'params': classification_head.parameters(), 'lr': args.lr * args.head_lr_factor})

    # set-up scheduler
    scheduler = ExponentialLR(optimizer, gamma=args.gamma_lr)

    # define loss
    loss_fn = nn.CrossEntropyLoss()

    def collator(input):
        output = {'sources': [], 'contexts': [],
                    'targets': []}
        for index, item in enumerate(input):
            if item['split'] == 'train':
                neg_samples = args.neg_samples_train
            else:
                neg_samples = args.neg_samples_eval
            source_input = [item['source_title'], item['source_abstract']]
            if args.insert_section:
                context_input = [item['target_section_title_gt']] #check
            else:
                context_input = ['']
            # insert abstract keywords
            if args.insert_mentions:
                if context_input[0] != '':
                    context_input[0] += f'{tokenizer.sep_token}'
                context_input[0] += item['target_abstract_kw_gt']

            context_input.append(item['sentence_proximity_target_gt'])

            target_input = [
                f"{item['target_title']}", item['target_abstract']]
        
            output['sources'].append(source_input)
            output['contexts'].append(context_input)
            output['targets'].append(target_input)
            for i in range(neg_samples):
                link_context_neg = item[f"sentence_proximity_target_neg_{i}"]

                if args.insert_section:
                    source_section_neg = item[f"target_section_title_neg_{i}"]
                    context_input = [source_section_neg]
                else:
                    context_input = ['']
                # insert abstract keywords
                if args.insert_mentions:
                    if context_input[0] != '':
                        context_input[0] += f'{tokenizer.sep_token}'
                    context_input[0] += item[f"target_abstract_kw_neg_{i}"]

                context_input.append(link_context_neg)
                output['contexts'].append(context_input)

        for key in output:
            output[key] = tokenizer(output[key], padding='max_length', truncation=True,
                                        return_tensors='pt', max_length=args.max_tokens)
        return output

    logger.info("Loading datasets")
    train_set = WikiDataset(args.data_dir, 'train', args.neg_samples_train)
    val_set = WikiDataset(args.data_dir, 'val', args.neg_samples_eval)
    logger.info(f"Train set size: {len(train_set)}")
    logger.info(f"Validation set size: {len(val_set)}")

    logger.info("Creating dataloaders")
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              drop_last=True,
                              collate_fn=collator,
                              pin_memory=True)
    val_loader = DataLoader(val_set,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            drop_last=True,
                            collate_fn=collator,
                            pin_memory=True)

    

    if args.full_freeze_steps > 0:
        logger.info(
            f"Freezing all layers except classification head for {args.full_freeze_steps} steps")
        if args.split_models:
            for param in model_articles.parameters():
                param.requires_grad = False
            for param in model_contexts.parameters():
                param.requires_grad = False
            for param in classification_head.parameters():
                param.requires_grad = True
        else:
            for param in model.parameters():
                param.requires_grad = False
            for param in classification_head.parameters():
                param.requires_grad = True
    else:
        logger.info(f"Freezing first {args.freeze_layers} layers")
        if args.split_models:
            model_articles = freeze_model(model_articles, args.model_architecture, args.freeze_layers)
            model_contexts = freeze_model(model_contexts, args.model_architecture, args.freeze_layers)
        else:
            model = freeze_model(model, args.model_architecture, args.freeze_layers)

    # prepare all objects with accelerator
    classification_head, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        classification_head, optimizer, train_loader, val_loader, scheduler)
    if args.split_models:
        model_articles, model_contexts = accelerator.prepare(
            model_articles, model_contexts)
    else:
        model = accelerator.prepare(model)

    logger.info("Starting training")
    step = 0
    running_loss = 0
    for epoch in range(args.num_epochs):
        for index, data in enumerate(train_loader):
            step += 1
            # multiple forward passes accumulate gradients
            # source: https://discuss.pytorch.org/t/multiple-model-forward-followed-by-one-loss-backward/20868
            if args.split_models:
                output_source = model_articles(
                    **data['sources'])['last_hidden_state'][:, 0, :]
                output_target = model_articles(
                    **data['targets'])['last_hidden_state'][:, 0, :]
            else:
                output_source = model(
                    **data['sources'])['last_hidden_state'][:, 0, :]
                output_target = model(
                    **data['targets'])['last_hidden_state'][:, 0, :]

            if args.split_models:
                output_context = model_contexts(
                    **data['contexts'])['last_hidden_state'][:, 0, :]
            else:
                output_context = model(
                    **data['contexts'])['last_hidden_state'][:, 0, :]
            # output_source has shape (batch_size, hidden_size)
            # output_target has shape (batch_size, hidden_size)
            # output_context has shape(batch_size * (neg_samples + 1), hidden_size)
            # we want to produce a tensor of shape (batch_size * (neg_samples + 1), 1)
            # we need to expand output_source and output_target to match the shape of output_context
            output_source = output_source.repeat_interleave(
                args.neg_samples_train + 1, dim=0)
            output_target = output_target.repeat_interleave(
                args.neg_samples_train + 1, dim=0)
            embeddings = torch.cat([output_source,
                                    output_context,
                                    output_target], dim=1)
            logits = classification_head(embeddings)
            # logits has shape (batch_size * (neg_samples + 1), 1)
            # we need to reshape it to (batch_size, neg_samples + 1)
            logits = logits.view(-1, args.neg_samples_train + 1)
            labels = torch.zeros_like(logits)
            labels[:, 0] = 1
            # shuffle logits and labels
            indices = torch.randperm(logits.shape[1])
            logits = logits[:, indices]
            labels = labels[:, indices]
            loss = loss_fn(
                logits / args.temperature, labels) / args.ga_steps
            accelerator.backward(loss)
            if (index + 1) % args.ga_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            # save running loss
            running_loss += loss.item() * args.ga_steps
            # unfreeze model if necessary
            if step == args.full_freeze_steps:
                logger.info(
                    f"Unfreezing model except first {args.freeze_layers} layers")
                if args.split_models:
                    model_articles = accelerator.unwrap_model(model_articles)
                    model_contexts = accelerator.unwrap_model(model_contexts)
                    model_contexts = unfreeze_model(model_contexts, args.model_architecture, args.freeze_layers)
                    model_articles = unfreeze_model(model_articles, args.model_architecture, args.freeze_layers)
                    model_articles = accelerator.prepare(model_articles)
                    model_contexts = accelerator.prepare(model_contexts)
                else:
                    model = accelerator.unwrap_model(model)
                    model = unfreeze_model(model, args.model_architecture, args.freeze_layers)
                    model = accelerator.prepare(model)

            # print loss
            if step % args.print_steps == 0:
                logger.info(
                    f"Step {step}: loss = {running_loss / args.print_steps}")
                if accelerator.is_main_process:
                    writer.add_scalar(
                        'train/loss', running_loss / args.print_steps, step)
                running_loss = 0

            if step % args.scheduler_steps == 0:
                logger.info(f"Step {step}: scheduler step")
                scheduler.step()
                logger.info(
                    f"Encoders learning rate: {scheduler.get_last_lr()[0]}")
                if args.split_models:
                    logger.info(
                        f"Classification head learning rate: {scheduler.get_last_lr()[2]}")
                else:
                    logger.info(
                        f"Classification head learning rate: {scheduler.get_last_lr()[1]}")

            # save model
            if step % args.save_steps == 0:
                logger.info(f"Step {step}: saving model")
                accelerator.wait_for_everyone()
                # accelerator needs to unwrap model and classification head
                if args.split_models:
                    accelerator.unwrap_model(model_articles).save_pretrained(os.path.join(
                        output_dir, f"model_articles_{step}"))
                    accelerator.unwrap_model(model_contexts).save_pretrained(os.path.join(
                        output_dir, f"model_contexts_{step}"))
                else:
                    accelerator.unwrap_model(model).save_pretrained(os.path.join(
                        output_dir, f"model_{step}"))
                torch.save(accelerator.unwrap_model(classification_head).state_dict(), os.path.join(
                    output_dir, f"classification_head_{step}.pth"))

            # evaluate model
            if step % args.eval_steps == 0:
                logger.info(f"Step {step}: evaluating model")
                if args.split_models:
                    model_articles.eval()
                    model_contexts.eval()
                else:
                    model.eval()
                with torch.no_grad():
                    true_pos = 0
                    true_neg = 0
                    false_pos = 0
                    false_neg = 0
                    mrr_at_k = {'1': 0, '5': 0, '10': 0, 'max': 0}
                    hits_at_k = {'1': 0, '5': 0, '10': 0, 'max': 0}
                    ndcg_at_k = {'1': 0, '5': 0, '10': 0, 'max': 0}
                    n_lists = 0
                    total = 0

                    running_val_loss = 0
                    for j, val_data in (pbar := tqdm(enumerate(val_loader), total=len(val_loader))):
                        if j % 20 == 0:
                            pbar.set_description(
                                f"True pos: {true_pos}, True neg: {true_neg}, False pos: {false_pos}, False neg: {false_neg}, Total: {total}")
                        if args.split_models:
                            output_source = model_articles(
                                **val_data['sources'])['last_hidden_state'][:, 0, :]
                            output_target = model_articles(
                                **val_data['targets'])['last_hidden_state'][:, 0, :]
                        else:
                            output_source = model(
                                **val_data['sources'])['last_hidden_state'][:, 0, :]
                            output_target = model(
                                **val_data['targets'])['last_hidden_state'][:, 0, :]

                        if args.split_models:
                            output_context = model_contexts(
                                **val_data['contexts'])['last_hidden_state'][:, 0, :]
                        else:
                            output_context = model(
                                **val_data['contexts'])['last_hidden_state'][:, 0, :]
                        output_source = output_source.repeat_interleave(
                            args.neg_samples_eval + 1, dim=0)
                        output_target = output_target.repeat_interleave(
                            args.neg_samples_eval + 1, dim=0)
                        embeddings = torch.cat([output_source,
                                                output_context,
                                                output_target], dim=1)
                        val_logits = classification_head(embeddings)
                        val_logits = val_logits.view(-1,
                                                     args.neg_samples_eval + 1)
                        labels = torch.zeros_like(val_logits)
                        labels[:, 0] = 1
                        val_loss = loss_fn(val_logits, labels)

                        # gather the results from all processes
                        val_logits = accelerator.pad_across_processes(
                            val_logits, dim=0, pad_index=-1)
                        labels = accelerator.pad_across_processes(
                            labels, dim=0, pad_index=-1)

                        val_logits = accelerator.gather_for_metrics(
                            val_logits).to('cpu')
                        labels = accelerator.gather_for_metrics(
                            labels).to('cpu')

                        val_loss = accelerator.gather_for_metrics(
                            val_loss).to('cpu')
                        running_val_loss += val_loss.mean().item()

                        n_lists += len(labels)

                        # calculate mrr, hits@k, ndcg@k
                        # sort probabilities in descending order and labels accordingly
                        val_logits, indices = torch.sort(
                            val_logits, dim=1, descending=True)
                        labels = torch.gather(labels, dim=1, index=indices)
                        # calculate mrr
                        for k in [1, 5, 10]:
                            mrr_at_k[str(k)] += torch.sum(
                                1 / (torch.nonzero(labels[:, :k])[:, 1].float() + 1)).item()
                        mrr_at_k['max'] += torch.sum(
                            1 / (torch.nonzero(labels)[:, 1].float() + 1)).item()

                        # calculate hits@k
                        for k in [1, 5, 10]:
                            hits_at_k[str(k)] += torch.sum(
                                torch.sum(labels[:, :k], dim=1)).item()
                        hits_at_k['max'] += torch.sum(
                            torch.sum(labels, dim=1)).item()

                        # calculate ndcg@k
                        for k in [1, 5, 10]:
                            ndcg_at_k[str(k)] += torch.sum(
                                torch.sum(labels[:, :k] / torch.log2(torch.arange(2, k + 2).float()), dim=1)).item()
                        ndcg_at_k['max'] += torch.sum(
                            torch.sum(labels / torch.log2(torch.arange(2, labels.shape[1] + 2).float()), dim=1)).item()

                      
                        probs = torch.softmax(val_logits, dim=1)
                        # predictions are 1 at the index with the highest probability, and 0 otherwise
                        preds = (probs == torch.max(
                            probs, dim=1, keepdim=True)[0]).long()

                        true_pos += torch.sum((preds == 1)
                                              & (labels == 1)).item()
                        true_neg += torch.sum((preds == 0)
                                              & (labels == 0)).item()
                        false_pos += torch.sum((preds == 1)
                                               & (labels == 0)).item()
                        false_neg += torch.sum((preds == 0)
                                               & (labels == 1)).item()
                        total = true_pos + true_neg + false_pos + false_neg

                        if j == len(val_loader) - 1:
                            pbar.set_description(
                                f"True pos: {true_pos}, True neg: {true_neg}, False pos: {false_pos}, False neg: {false_neg}, Total: {total}")
                    # calculate accuracy, precision, recall, f1 score
                    total = true_pos + true_neg + false_pos + false_neg
                    accuracy = (true_pos + true_neg) / total
                    precision = true_pos / \
                        (true_pos + false_pos) if true_pos + false_pos > 0 else 0
                    recall = true_pos / \
                        (true_pos + false_neg) if true_pos + false_neg > 0 else 0
                    f1 = 2 * precision * recall / \
                        (precision + recall) if precision + recall > 0 else 0
                    mrr_at_k = {k: v / n_lists for k, v in mrr_at_k.items()}
                    hits_at_k = {k: v / n_lists for k, v in hits_at_k.items()}
                    ndcg_at_k = {k: v / n_lists for k, v in ndcg_at_k.items()}
                    running_val_loss /= len(val_loader)

                    for k in ['1', '5', '10', 'max']:
                        logger.info(f"MRR@{k}: {mrr_at_k[k]}")
                        logger.info(f"Hits@{k}: {hits_at_k[k]}")
                        logger.info(f"NDCG@{k}: {ndcg_at_k[k]}")
                    logger.info(f"Accuracy: {accuracy}")
                    logger.info(f"Precision: {precision}")
                    logger.info(f"Recall: {recall}")
                    logger.info(f"F1: {f1}")
                    logger.info(f"Validation loss: {running_val_loss}")

                    if accelerator.is_main_process:
                        for k in ['1', '5', '10', 'max']:
                            writer.add_scalar(
                                f'val/mrr@{k}', mrr_at_k[k], step)
                            writer.add_scalar(
                                f'val/hits@{k}', hits_at_k[k], step)
                            writer.add_scalar(
                                f'val/ndcg@{k}', ndcg_at_k[k], step)
                        writer.add_scalar('val/accuracy', accuracy, step)
                        writer.add_scalar('val/precision', precision, step)
                        writer.add_scalar('val/recall', recall, step)
                        writer.add_scalar('val/f1', f1, step)
                        writer.add_scalar('val/loss', running_val_loss, step)
               
                if args.split_models:
                    model_articles.train()
                    model_contexts.train()
                else:
                    model.train()
                torch.cuda.empty_cache()
                gc.collect()

    logger.info("Training finished")
    if accelerator.is_main_process:
        writer.close()

    accelerator.wait_for_everyone()
    # save last version of models
    if args.split_models:
        accelerator.unwrap_model(model_articles).save_pretrained(os.path.join(
            output_dir, f"model_articles"))
        accelerator.unwrap_model(model_contexts).save_pretrained(os.path.join(
            output_dir, f"model_contexts"))
    else:
        accelerator.unwrap_model(model).save_pretrained(os.path.join(
            output_dir, f"model"))
    torch.save(accelerator.unwrap_model(classification_head).state_dict(), os.path.join(
        output_dir, f"classification_head.pth"))