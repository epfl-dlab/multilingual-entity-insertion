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
import nltk
from nltk import sent_tokenize
from urllib.parse import unquote
import json
from ast import literal_eval

multiprocess.set_start_method("spawn", force=True)
nltk.download('punkt', download_dir='/dlabdata1/tsoares/nltk_data')
nltk.data.path.append('/dlabdata1/tsoares/nltk_data')


def mask_negative_contexts(context, probs, backlog):
    sentences = [sentence.strip()
                 for sentence in sent_tokenize(context) if sentence.strip()]
    strategies = ['mask_span', 'mask_sentence', 'mask_word', 'no_mask']
    valid_strategies = ['mask_span', 'mask_sentence', 'mask_word', 'no_mask']
    if len(sentences) <= 2:
        valid_strategies.remove('mask_span')
    if len(sentences) == 1:
        valid_strategies.remove('mask_sentence')
    words = []
    for sentence in sentences:
        words.extend([word for word in sentence.replace(
            '\n', ' ').split() if word.strip()])
    if len(words) == 1:
        valid_strategies.remove('mask_word')

    mask_strategy = None
    for strategy in valid_strategies:
        if backlog[strategy] > 0:
            mask_strategy = strategy
            backlog[strategy] -= 1
            break
    if mask_strategy is None:
        if valid_strategies == ['no_mask']:
            probs['no_mask'] = 1
        mask_strategy = random.choices(
            strategies, weights=[probs[strategy] for strategy in strategies], k=1)[0]
        if mask_strategy not in valid_strategies:
            backlog[mask_strategy] += 1
            mask_strategy = random.choices(valid_strategies, weights=[
                                           probs[strategy] for strategy in valid_strategies], k=1)[0]
    if mask_strategy == 'no_mask':
        return context
    if mask_strategy == 'mask_word':
        sentence_index = random.randint(0, len(sentences) - 1)
        words = sentences[sentence_index].split(' ')
        mask_index = random.randint(0, len(words) - 1)
        masked_context = ''
        for i, sentence in enumerate(sentences):
            if i != sentence_index:
                masked_context += sentence + ' '
            else:
                masked_context += " ".join(
                    [word for j, word in enumerate(words) if j != mask_index]) + ' '
        return masked_context[:-1]

    if mask_strategy == 'mask_sentence':
        mask_index = random.randint(0, len(sentences) - 1)
        return " ".join(sentences[:mask_index]) + " " + " ".join(sentences[mask_index+1:])
    if mask_strategy == 'mask_span':
        mask_length = random.randint(2, min(len(sentences) - 1, 5))
        start_index = random.randint(0, len(sentences) - mask_length)
        return " ".join(sentences[:start_index]) + " " + " ".join(sentences[start_index + mask_length:])


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
    parser.add_argument('--model_architecture', type=str, choices=[
                        'BERT', 'RoBERTa', 'T5'], default='BERT', help='Model architecture')
    parser.add_argument('--data_dir', type=str,
                        required=True, help='Data directory')
    parser.add_argument('--data_dir_2', type=str, default='',
                        help='Data directory for second stage')
    parser.add_argument('--num_epochs', nargs='+', type=int,
                        default=[1], help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of workers for dataloader')
    parser.add_argument('--lr', nargs='+', type=float,
                        default=[1e-5], help='Learning rate')
    parser.add_argument('--gamma_lr', nargs='+', type=float,
                        default=[0.9], help='Gamma for lr scheduler')
    parser.add_argument('--print_steps', nargs='+', type=int, default=[1_000],
                        help='Number of steps between printing loss')
    parser.add_argument('--save_steps', nargs='+', type=int, default=[5_000],
                        help='Number of steps between saving model')
    parser.add_argument('--eval_steps', nargs='+', type=int, default=[5_000],
                        help='Number of steps between evaluating model on validation set')
    parser.add_argument('--scheduler_steps', nargs='+', type=int, default=[10_000],
                        help='Number of steps between scheduler steps')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint (needs --checkpoint_dir)')
    parser.add_argument('--checkpoint_dir', type=str,
                        help='Directory with checkpoint to resume training from')
    parser.add_argument('--ga_steps', nargs='+', type=int, default=[1],
                        help='Number of steps for gradient accumulation')
    parser.add_argument('--full_freeze_steps', type=int, default=0,
                        help='Number of steps to freeze all layers except classification head')
    parser.add_argument('--freeze_layers', type=int,
                        default=2, help='Number of initial layers to freeze')
    parser.add_argument('--head_lr_factor', type=float,
                        default=1, help='Factor for learning rate of classification head')
    parser.add_argument('--no_mask_perc', type=float, default=0.4,
                        help='Percentage of examples to not mask')
    parser.add_argument('--mask_mention_perc', type=float, default=0.2,
                        help='Percentage of mentions to mask')
    parser.add_argument('--mask_sentence_perc', type=float, default=0.3,
                        help='Percentage of sentences to mask')
    parser.add_argument('--mask_span_perc', type=float, default=0.1,
                        help='Percentage of spans to mask')
    parser.add_argument('--max_tokens', type=int, default=256,
                        help='Maximum number of tokens')
    parser.add_argument('--neg_samples_train', nargs='+', type=int, default=[1],
                        help='Number of negative samples for training')
    parser.add_argument('--neg_samples_eval', nargs='+', type=int, default=[20],
                        help='Number of negative samples for evaluation')
    parser.add_argument('--temperature', nargs='+', type=float, default=[1],
                        help='Temperature for softmax')
    parser.add_argument('--insert_mentions', action='store_true',
                        help='Whether to insert target mentions')
    parser.add_argument('--insert_section', action='store_true',
                        help='Whether to insert section title')
    parser.add_argument('--mask_negatives', action='store_true',
                        help='Whether to apply masking to negative samples')
    parser.add_argument('--two_stage', action='store_true',
                        help='If set, use two-stage training')
    parser.add_argument('--min_steps', nargs='+', type=int,
                        default=[0], help='Minimum number of steps to train for')
    parser.add_argument('--use_current_links', action='store_true',
                        help='Whether to use current links')
    parser.set_defaults(resume=False, insert_section=False, mask_negatives=False,
                        two_stage=False, insert_mentions=False, use_current_links=False)

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

    # check if data_dir_2 exists if two_stage is set
    if args.two_stage and not os.path.exists(args.data_dir_2):
        raise ValueError(
            f"Data directory {args.data_dir_2} does not exist. It is required for two-stage training")

    # check if noise percentages add up to 1
    if abs(args.no_mask_perc + args.mask_mention_perc + args.mask_sentence_perc + args.mask_span_perc - 1) > 1e-5:
        raise ValueError(
            "Noise percentages do not add up to 1")
    weights = [args.mask_span_perc, args.mask_sentence_perc,
               args.mask_mention_perc, args.no_mask_perc]
    neg_map = {'easy_replace_source': 0, 'hard_replace_source': 1, 'easy_replace_target': 2,
               'hard_replace_target': 3, 'easy_replace_context': 4, 'hard_replace_context': 5}
    neg_map_rev = {value: key for key, value in neg_map.items()}
    noise_map = {'no_mask': 0, 'mask_mention': 1,
                 'mask_sentence': 2, 'mask_span': 3}
    noise_map_rev = {value: key for key, value in noise_map.items()}

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

    # if using two stage pipeline, make sure all relevant arguments have length 2
    # if they don't, repeat the first element
    for arg in vars(args):
        if isinstance(getattr(args, arg), list):
            if len(getattr(args, arg)) == 1:
                setattr(args, arg, getattr(args, arg) * 2)

    # log arguments
    logger.info("Arguments:")
    for arg in vars(args):
        logger.info(f"\t- {arg}: {getattr(args, arg)}")

    if args.resume:
        logger.info("Loading model")
        try:
            model = AutoModel.from_pretrained(args.checkpoint_dir, 'model')
            logger.info("Model loaded from checkpoint directory")
        except OSError:
            logger.info("Could not load model from checkpoint directory")
            logger.info("Initializing model from provided model name")
            model = AutoModel.from_pretrained(args.model_name)
        model_size = model.config.hidden_size
        try:
            classification_head = Sequential(nn.Linear(model_size, model_size),
                                             nn.ReLU(),
                                             nn.Linear(model_size, 1))
            classification_head.load_state_dict(torch.load(os.path.join(
                args.checkpoint_dir, 'classification_head.pth'), map_location='cpu'))
            logger.info("Classification head loaded from checkpoint directory")
        except OSError:
            logger.info(
                "Could not load classification head from checkpoint directory")
            logger.info("Initializing classification head with random weights")
            classification_head = Sequential(nn.Linear(model_size, model_size),
                                             nn.ReLU(),
                                             nn.Linear(model_size, 1))
    else:
        logger.info("Initializing model")
        model = AutoModel.from_pretrained(args.model_name)
        model_size = model.config.hidden_size

        classification_head = Sequential(nn.Linear(model_size, model_size),
                                         nn.ReLU(),
                                         nn.Linear(model_size, 1))

    # Remove decoder layers if present
    if args.model_architecture == 'T5':
        model = model.encoder

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.save_pretrained(os.path.join(output_dir, 'tokenizer'))

    logger.info("Initializing optimizer")
    optimizer = optim.Adam(model.parameters(), lr=args.lr[0])
    # add classification head to optimizer
    optimizer.add_param_group(
        {'params': classification_head.parameters(), 'lr': args.lr[0] * args.head_lr_factor})

    # set-up scheduler
    scheduler = ExponentialLR(optimizer, gamma=args.gamma_lr[0])

    # define loss
    loss_fn = nn.CrossEntropyLoss()

    noise_backlog = {'no_mask': 0, 'mask_mention': 0,
                     'mask_sentence': 0, 'mask_span': 0}
    neg_noise_backlog = {'no_mask': 0, 'mask_word': 0,
                         'mask_sentence': 0, 'mask_span': 0}

    def collator(input):
        output = {'contexts': [], 'noises': []}
        if input[0]['split'] == 'train':
            noise_types = ['mask_span', 'mask_sentence',
                           'mask_mention', 'no_mask']
            for index, item in enumerate(input):
                if item['target_title'] not in mention_map:
                    mention_map[item['target_title']] = ''
                found = False
                if (item['link_context'][:item['context_span_start_index']] + item['link_context'][item['context_span_end_index']:]).strip() != '':
                    if item['context_span_start_index'] <= item['context_sentence_start_index'] and item['context_span_end_index'] >= item['context_sentence_end_index']:
                        valid_noise_types = [
                            'mask_span', 'mask_sentence', 'mask_mention', 'no_mask']
                        found = True
                if not found and (item['link_context'][:item['context_sentence_start_index']] + item['link_context'][item['context_sentence_end_index']:]).strip() != '':
                    if item['context_sentence_start_index'] <= item['context_mention_start_index'] and item['context_sentence_end_index'] > item['context_mention_end_index'] + 1:
                        valid_noise_types = ['mask_sentence',
                                             'mask_mention', 'no_mask']
                        found = True
                if not found and (item['link_context'][:item['context_mention_start_index']] + item['link_context'][item['context_mention_end_index']:]).strip() != '':
                    valid_noise_types = ['mask_mention', 'no_mask']
                    found = True
                if not found:
                    valid_noise_types = ['no_mask']

                noise_type = None
                for category in noise_types:
                    if noise_backlog[category] > 0 and category in valid_noise_types:
                        noise_type = category
                        noise_backlog[category] -= 1
                        break
                if noise_type is None:
                    noise_type = random.choices(
                        noise_types, weights=weights, k=1)[0]
                    if noise_type not in valid_noise_types:
                        noise_backlog[noise_type] += 1
                        noise_type = random.choices(
                            valid_noise_types, weights=weights[-len(valid_noise_types):], k=1)[0]

                if noise_type == 'mask_span':
                    item['link_context'] = item['link_context'][:int(
                        item['context_span_start_index'])] + item['link_context'][int(item['context_span_end_index']):]
                elif noise_type == 'mask_sentence':
                    item['link_context'] = item['link_context'][:int(
                        item['context_sentence_start_index'])] + item['link_context'][int(item['context_sentence_end_index']):]
                elif noise_type == 'mask_mention':
                    item['link_context'] = item['link_context'][:int(
                        item['context_mention_start_index'])] + item['link_context'][int(item['context_mention_end_index']):]
                item['link_context'] = re.sub(
                    ' +', ' ', item['link_context'])
                item['link_context'] = re.sub(
                    '\n ', '\n', item['link_context'])
                item['link_context'] = re.sub(
                    '\n+', '\n', item['link_context'])
                item['link_context'] = item['link_context'].strip()

                if args.insert_mentions:
                    context_input = [
                        f"{item['target_title']} {mention_map[item['target_title']]}{tokenizer.sep_token}{item['target_lead']}"]
                else:
                    context_input = [
                        f"{item['target_title']}{tokenizer.sep_token}{item['target_lead']}"]
                if args.insert_section:
                    context_input.append(
                        f"{item['source_section']}{tokenizer.sep_token}{item['link_context']}")
                else:
                    context_input.append(item['link_context'])

                output['noises'].append(noise_map[noise_type])
                output['contexts'].append(context_input)

                mask_probs = {'no_mask': args.no_mask_perc, 'mask_word': args.mask_mention_perc,
                              'mask_sentence': args.mask_sentence_perc, 'mask_span': args.mask_span_perc}
                for i in range(args.neg_samples_train[0]):
                    link_context_neg = item[f"link_context_neg_{i}"]
                    source_section_neg = item[f"source_section_neg_{i}"]
                    if args.mask_negatives:
                        link_context_neg = mask_negative_contexts(
                            link_context_neg, mask_probs, neg_noise_backlog)

                    if args.insert_mentions:
                        context_input = [
                            f"{item['target_title']} {mention_map[item['target_title']]}{tokenizer.sep_token}{item['target_lead']}"]
                    else:
                        context_input = [
                            f"{item['target_title']}{tokenizer.sep_token}{item['target_lead']}"]
                    if args.insert_section:
                        context_input.append(
                            f"{source_section_neg}{tokenizer.sep_token}{link_context_neg}")
                    else:
                        context_input.append(link_context_neg)
                    output['contexts'].append(context_input)
        else:
            for index, item in enumerate(input):
                if item['target_title'] not in mention_map:
                    mention_map[item['target_title']] = ''

                if args.insert_mentions:
                    context_input = [
                        f"{item['target_title']} {mention_map[item['target_title']]}{tokenizer.sep_token}{item['target_lead']}"]
                else:
                    context_input = [
                        f"{item['target_title']}{tokenizer.sep_token}{item['target_lead']}"]
                if args.insert_section:
                    context_input.append(
                        f"{item['source_section']}{tokenizer.sep_token}{item['link_context']}")
                else:
                    context_input.append(item['link_context'])

                output['noises'].append(noise_map[item['noise_strategy']])
                output['contexts'].append(context_input)
                mask_probs = {'no_mask': args.no_mask_perc, 'mask_word': args.mask_mention_perc,
                              'mask_sentence': args.mask_sentence_perc, 'mask_span': args.mask_span_perc}
                for i in range(args.neg_samples_eval[0]):
                    link_context_neg = item[f"link_context_neg_{i}"]
                    source_section_neg = item[f"source_section_neg_{i}"]
                    if args.mask_negatives:
                        link_context_neg = mask_negative_contexts(
                            link_context_neg, mask_probs, neg_noise_backlog)

                    if args.insert_mentions:
                        context_input = [
                            f"{item['target_title']} {mention_map[item['target_title']]}{tokenizer.sep_token}{item['target_lead']}"]
                    else:
                        context_input = [
                            f"{item['target_title']}{tokenizer.sep_token}{item['target_lead']}"]
                    if args.insert_section:
                        context_input.append(
                            f"{source_section_neg}{tokenizer.sep_token}{link_context_neg}")
                    else:
                        context_input.append(link_context_neg)
                    output['contexts'].append(context_input)

        for key in output:
            if key in ['noises', 'current_links_supindex', 'current_links_subindex']:
                output[key] = torch.tensor(output[key])
            else:
                output[key] = tokenizer(output[key], padding='max_length', truncation=True,
                                        return_tensors='pt', max_length=args.max_tokens)
        return output

    logger.info("Loading datasets")
    train_set = WikiDataset(args.data_dir, 'train', args.neg_samples_train[0])
    val_set = WikiDataset(args.data_dir, 'val', args.neg_samples_eval[0])
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

    logger.info("Loading mention knowledge")
    mentions = pd.read_parquet(os.path.join(
        args.data_dir, 'mentions.parquet')).to_dict('records')
    mention_map = {}
    for mention in mentions:
        target_title = unquote(mention['target_title']).replace('_', ' ')
        if target_title in mention_map:
            mention_map[target_title].append(mention['mention'])
        else:
            mention_map[target_title] = [mention['mention']]

    for mention in mention_map:
        # only keep a random subset of 10 mentions
        if len(mention_map[mention]) > 10:
            mention_map[mention] = random.sample(mention_map[mention], 10)
        mention_map[mention] = ' '.join(mention_map[mention])

    if args.full_freeze_steps > 0:
        logger.info(
            f"Freezing all layers except classification head for {args.full_freeze_steps} steps")
        for param in model.parameters():
            param.requires_grad = False
        for param in classification_head.parameters():
            param.requires_grad = True
    else:
        logger.info(f"Freezing first {args.freeze_layers} layers")
        model = freeze_model(
            model, args.model_architecture, args.freeze_layers)

    # prepare all objects with accelerator
    model, classification_head, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, classification_head, optimizer, train_loader, val_loader, scheduler)

    logger.info("Starting training")
    step = 0
    running_loss = 0
    epoch = 0
    while epoch < args.num_epochs[0] or step < args.min_steps[0]:
        epoch += 1
        for index, data in enumerate(train_loader):
            step += 1
            # multiple forward passes accumulate gradients
            # source: https://discuss.pytorch.org/t/multiple-model-forward-followed-by-one-loss-backward/20868
            embeddings = model(**data['contexts']
                               )['last_hidden_state'][:, 0, :]
            logits = classification_head(embeddings)
            # logits has shape (batch_size * (neg_samples + 1), 1)
            # we need to reshape it to (batch_size, neg_samples + 1)
            logits = logits.view(-1, args.neg_samples_train[0] + 1)
            labels = torch.zeros_like(logits)
            labels[:, 0] = 1
            # shuffle logits and labels
            indices = torch.randperm(logits.shape[1])
            logits = logits[:, indices]
            labels = labels[:, indices]
            loss = loss_fn(
                logits / args.temperature[0], labels) / args.ga_steps[0]
            accelerator.backward(loss)
            if (index + 1) % args.ga_steps[0] == 0:
                optimizer.step()
                optimizer.zero_grad()
            # save running loss
            running_loss += loss.item() * args.ga_steps[0]
            # unfreeze model if necessary
            if step == args.full_freeze_steps:
                logger.info(
                    f"Unfreezing model except first {args.freeze_layers} layers")
                model = accelerator.unwrap_model(model)
                model = unfreeze_model(
                    model, args.model_architecture, args.freeze_layers)
                model = accelerator.prepare(model)

            # print loss
            if step % args.print_steps[0] == 0:
                logger.info(
                    f"Step {step}: loss = {running_loss / args.print_steps[0]}")
                if accelerator.is_main_process:
                    writer.add_scalar(
                        'train/loss', running_loss / args.print_steps[0], step)
                running_loss = 0

            if step % args.scheduler_steps[0] == 0:
                logger.info(f"Step {step}: scheduler step")
                scheduler.step()
                logger.info(
                    f"Encoders learning rate: {scheduler.get_last_lr()[0]}")
                logger.info(
                    f"Classification head learning rate: {scheduler.get_last_lr()[1]}")

            # save model
            if step % args.save_steps[0] == 0:
                logger.info(f"Step {step}: saving model")
                accelerator.wait_for_everyone()
                # accelerator needs to unwrap model and classification head
                accelerator.unwrap_model(model).save_pretrained(os.path.join(
                    output_dir, f"model_{step}"))
                torch.save(accelerator.unwrap_model(classification_head).state_dict(), os.path.join(
                    output_dir, f"classification_head_{step}.pth"))

            # evaluate model
            if step % args.eval_steps[0] == 0:
                logger.info(f"Step {step}: evaluating model")
                model.eval()
                with torch.no_grad():
                    true_pos = 0
                    true_neg = 0
                    false_pos = 0
                    false_neg = 0
                    mrr_at_k = {'1': 0, '5': 0, '10': 0, 'max': 0}
                    hits_at_k = {'1': 0, '5': 0, '10': 0, 'max': 0}
                    ndcg_at_k = {'1': 0, '5': 0, '10': 0, 'max': 0}
                    noise_perf = {i: {'mrr': {'1': 0, '5': 0, '10': 0, 'max': 0},
                                      'hits': {'1': 0, '5': 0, '10': 0, 'max': 0},
                                      'ndcg': {'1': 0, '5': 0, '10': 0, 'max': 0},
                                      'n_lists': 0} for i in range(len(noise_map))}
                    n_lists = 0
                    total = 0

                    running_val_loss = 0
                    for j, val_data in (pbar := tqdm(enumerate(val_loader), total=len(val_loader))):
                        if j % 20 == 0:
                            pbar.set_description(
                                f"True pos: {true_pos}, True neg: {true_neg}, False pos: {false_pos}, False neg: {false_neg}, Total: {total}")
                        embeddings = model(
                            **val_data['contexts'])['last_hidden_state'][:, 0, :]
                        val_logits = classification_head(embeddings)
                        val_logits = val_logits.view(-1,
                                                     args.neg_samples_eval[0] + 1)
                        labels = torch.zeros_like(val_logits)
                        labels[:, 0] = 1
                        val_loss = loss_fn(val_logits, labels)

                        # gather the results from all processes
                        val_logits = accelerator.pad_across_processes(
                            val_logits, dim=0, pad_index=-1)
                        labels = accelerator.pad_across_processes(
                            labels, dim=0, pad_index=-1)
                        noise = accelerator.pad_across_processes(
                            val_data['noises'], dim=0, pad_index=-1)

                        val_logits = accelerator.gather_for_metrics(
                            val_logits).to('cpu')
                        labels = accelerator.gather_for_metrics(
                            labels).to('cpu')
                        noise = accelerator.gather_for_metrics(
                            noise).to('cpu')

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

                        # compute discritized scores for each noise type
                        for i in range(len(noise_map)):
                            noise_part = noise == i
                            labels_part = labels[noise_part]

                            for k in [1, 5, 10]:
                                noise_perf[i]['mrr'][str(k)] += torch.sum(
                                    1 / (torch.nonzero(labels_part[:, :k])[:, 1].float() + 1)).item()
                            noise_perf[i]['mrr']['max'] += torch.sum(
                                1 / (torch.nonzero(labels_part)[:, 1].float() + 1)).item()

                            for k in [1, 5, 10]:
                                noise_perf[i]['hits'][str(k)] += torch.sum(
                                    torch.sum(labels_part[:, :k], dim=1)).item()
                            noise_perf[i]['hits']['max'] += torch.sum(
                                torch.sum(labels_part, dim=1)).item()

                            for k in [1, 5, 10]:
                                noise_perf[i]['ndcg'][str(k)] += torch.sum(
                                    torch.sum(labels_part[:, :k] / torch.log2(torch.arange(2, k + 2).float()), dim=1)).item()
                            noise_perf[i]['ndcg']['max'] += torch.sum(
                                torch.sum(labels_part / torch.log2(torch.arange(2, labels_part.shape[1] + 2).float()), dim=1)).item()

                            noise_perf[i]['n_lists'] += len(labels_part)

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
                    for i in range(len(noise_map)):
                        if noise_perf[i]['n_lists'] > 0:
                            noise_perf[i]['mrr'] = {
                                k: v / noise_perf[i]['n_lists'] for k, v in noise_perf[i]['mrr'].items()}
                            noise_perf[i]['hits'] = {
                                k: v / noise_perf[i]['n_lists'] for k, v in noise_perf[i]['hits'].items()}
                            noise_perf[i]['ndcg'] = {
                                k: v / noise_perf[i]['n_lists'] for k, v in noise_perf[i]['ndcg'].items()}
                    running_val_loss /= len(val_loader)

                    for k in ['1', '5', '10', 'max']:
                        logger.info(f"MRR@{k}: {mrr_at_k[k]}")
                        logger.info(f"Hits@{k}: {hits_at_k[k]}")
                        logger.info(f"NDCG@{k}: {ndcg_at_k[k]}")
                    for i in range(len(noise_map)):
                        if noise_perf[i]['n_lists'] > 0:
                            logger.info(f"Noise strategy {noise_map_rev[i]}:")
                            for k in ['1', '5', '10', 'max']:
                                logger.info(
                                    f"\t- MRR@{k}: {noise_perf[i]['mrr'][k]}")
                                logger.info(
                                    f"\t- Hits@{k}: {noise_perf[i]['hits'][k]}")
                                logger.info(
                                    f"\t- NDCG@{k}: {noise_perf[i]['ndcg'][k]}")
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
                        for i in range(len(noise_map)):
                            if noise_perf[i]['n_lists'] > 0:
                                for k in ['1', '5', '10', 'max']:
                                    writer.add_scalar(
                                        f'val_noise/{noise_map_rev[i]}_mrr@{k}', noise_perf[i]['mrr'][k], step)
                                    writer.add_scalar(
                                        f'val_noise/{noise_map_rev[i]}_hits@{k}', noise_perf[i]['hits'][k], step)
                                    writer.add_scalar(
                                        f'val_noise/{noise_map_rev[i]}_ndcg@{k}', noise_perf[i]['ndcg'][k], step)

                model.train()
                torch.cuda.empty_cache()
                gc.collect()

    if not args.two_stage:
        logger.info("Training finished")
        if accelerator.is_main_process:
            writer.close()

        accelerator.wait_for_everyone()
        # save last version of models
        accelerator.unwrap_model(model).save_pretrained(os.path.join(
            output_dir, f"model"))
        torch.save(accelerator.unwrap_model(classification_head).state_dict(), os.path.join(
            output_dir, f"classification_head.pth"))
        # exit script
        sys.exit()

    logger.info("Starting second stage of training")
    # delete all unnecessary objects
    del train_set, val_set, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()

    missing_map = {'present': 0, 'missing_mention': 1,
                   'missing_sentence': 2, 'missing_span': 3}
    missing_map_rev = {value: key for key, value in noise_map.items()}

    def simple_collator(input):
        output = {'contexts': [], 'noises': []}
        if input[0]['split'] == 'train':
            for index, item in enumerate(input):
                if item['target_title'] not in mention_map:
                    mention_map[item['target_title']] = ''

                if args.insert_mentions:
                    context_input = [
                        f"{item['target_title']} {mention_map[item['target_title']]}{tokenizer.sep_token}{item['target_lead']}"]
                else:
                    context_input = [
                        f"{item['target_title']}{tokenizer.sep_token}{item['target_lead']}"]
                if args.insert_section:
                    context_input.append(
                        f"{item['source_section']}{tokenizer.sep_token}{item['link_context']}")
                else:
                    context_input.append(item['link_context'])
                output['contexts'].append(context_input)

                for i in range(args.neg_samples_train[1]):
                    link_context_neg = item[f"link_context_neg_{i}"]
                    source_section_neg = item[f"source_section_neg_{i}"]
                    if args.insert_mentions:
                        context_input = [
                            f"{item['target_title']} {mention_map[item['target_title']]}{tokenizer.sep_token}{item['target_lead']}"]
                    else:
                        context_input = [
                            f"{item['target_title']}{tokenizer.sep_token}{item['target_lead']}"]
                    if args.insert_section:
                        context_input.append(
                            f"{source_section_neg}{tokenizer.sep_token}{link_context_neg}")
                    else:
                        context_input.append(link_context_neg)
                    output['contexts'].append(context_input)
        else:
            for index, item in enumerate(input):
                if item['target_title'] not in mention_map:
                    mention_map[item['target_title']] = ''
                if args.insert_mentions:
                    context_input = [
                        f"{item['target_title']} {mention_map[item['target_title']]}{tokenizer.sep_token}{item['target_lead']}"]
                else:
                    context_input = [
                        f"{item['target_title']}{tokenizer.sep_token}{item['target_lead']}"]
                if args.insert_section:
                    context_input.append(
                        f"{item['source_section']}{tokenizer.sep_token}{item['link_context']}")
                else:
                    context_input.append(item['link_context'])

                output['noises'].append(missing_map[item['missing_category']])
                output['contexts'].append(context_input)
                for i in range(args.neg_samples_eval[1]):
                    link_context_neg = item[f"link_context_neg_{i}"]
                    source_section_neg = item[f"source_section_neg_{i}"]
                    if args.insert_mentions:
                        context_input = [
                            f"{item['target_title']} {mention_map[item['target_title']]}{tokenizer.sep_token}{item['target_lead']}"]
                    else:
                        context_input = [
                            f"{item['target_title']}{tokenizer.sep_token}{item['target_lead']}"]
                    if args.insert_section:
                        context_input.append(
                            f"{source_section_neg}{tokenizer.sep_token}{link_context_neg}")
                    else:
                        context_input.append(link_context_neg)
                    output['contexts'].append(context_input)

        for key in output:
            if key in ['noises', 'current_links_supindex', 'current_links_subindex']:
                output[key] = torch.tensor(output[key])
            else:
                output[key] = tokenizer(output[key], padding='max_length',
                                        truncation=True, return_tensors='pt', max_length=args.max_tokens)
        return output

    logger.info("Loading datasets")
    train_set = WikiDataset(args.data_dir_2, 'train',
                            args.neg_samples_train[1])
    val_set = WikiDataset(args.data_dir_2, 'val', args.neg_samples_eval[1])
    logger.info(f"Train set size: {len(train_set)}")
    logger.info(f"Validation set size: {len(val_set)}")

    logger.info("Creating dataloaders")
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              drop_last=True,
                              collate_fn=simple_collator,
                              pin_memory=True)
    val_loader = DataLoader(val_set,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            drop_last=True,
                            collate_fn=simple_collator,
                            pin_memory=True)

    train_loader, val_loader = accelerator.prepare(train_loader, val_loader)

    logger.info("Loading mention knowledge")
    mentions = pd.read_parquet(os.path.join(
        args.data_dir_2, 'mentions.parquet')).to_dict('records')
    mention_map = {}
    for mention in mentions:
        target_title = unquote(mention['target_title']).replace('_', ' ')
        if target_title in mention_map:
            mention_map[target_title].append(mention['mention'])
        else:
            mention_map[target_title] = [mention['mention']]

    for mention in mention_map:
        # only keep a random subset of 10 mentions
        if len(mention_map[mention]) > 10:
            mention_map[mention] = random.sample(mention_map[mention], 10)
        mention_map[mention] = ' '.join(mention_map[mention])

    logger.info("Starting training")
    running_loss = 0
    new_steps = step
    epoch = 0
    while epoch < args.num_epochs[1] or (step - new_steps) < args.min_steps[1]:
        epoch += 1
        for index, data in enumerate(train_loader):
            step += 1
            # multiple forward passes accumulate gradients
            # source: https://discuss.pytorch.org/t/multiple-model-forward-followed-by-one-loss-backward/20868
            embeddings = model(**data['contexts']
                               )['last_hidden_state'][:, 0, :]
            logits = classification_head(embeddings)
            logits = logits.view(-1, args.neg_samples_train[1] + 1)
            labels = torch.zeros_like(logits)
            labels[:, 0] = 1
            loss = loss_fn(
                logits / args.temperature[1], labels) / args.ga_steps[1]
            accelerator.backward(loss)
            if (index + 1) % args.ga_steps[1] == 0:
                optimizer.step()
                optimizer.zero_grad()
            # save running loss
            running_loss += loss.item() * args.ga_steps[1]
            # print loss
            if step % args.print_steps[1] == 0:
                logger.info(
                    f"Step {step}: loss = {running_loss / args.print_steps[1]}")
                if accelerator.is_main_process:
                    writer.add_scalar(
                        'train/loss', running_loss / args.print_steps[1], step)
                running_loss = 0

            if step % args.scheduler_steps[1] == 0:
                logger.info(f"Step {step}: scheduler step")
                scheduler.step()
                logger.info(
                    f"Encoders learning rate: {scheduler.get_last_lr()[0]}")
                logger.info(
                    f"Classification head learning rate: {scheduler.get_last_lr()[1]}")

            # save model
            if step % args.save_steps[1] == 0:
                logger.info(f"Step {step}: saving model")
                accelerator.wait_for_everyone()
                # accelerator needs to unwrap model and classification head
                accelerator.unwrap_model(model).save_pretrained(os.path.join(
                    output_dir, f"model_{step}"))
                torch.save(accelerator.unwrap_model(classification_head).state_dict(), os.path.join(
                    output_dir, f"classification_head_{step}.pth"))

            # evaluate model
            if step % args.eval_steps[1] == 0:
                logger.info(f"Step {step}: evaluating model")
                model.eval()
                with torch.no_grad():
                    true_pos = 0
                    true_neg = 0
                    false_pos = 0
                    false_neg = 0
                    mrr_at_k = {'1': 0, '5': 0, '10': 0, 'max': 0}
                    hits_at_k = {'1': 0, '5': 0, '10': 0, 'max': 0}
                    ndcg_at_k = {'1': 0, '5': 0, '10': 0, 'max': 0}
                    noise_perf = {i: {'mrr': {'1': 0, '5': 0, '10': 0, 'max': 0},
                                      'hits': {'1': 0, '5': 0, '10': 0, 'max': 0},
                                      'ndcg': {'1': 0, '5': 0, '10': 0, 'max': 0},
                                      'n_lists': 0} for i in range(len(missing_map))}
                    n_lists = 0
                    total = 0

                    running_val_loss = 0
                    for j, val_data in (pbar := tqdm(enumerate(val_loader), total=len(val_loader))):
                        if j % 20 == 0:
                            pbar.set_description(
                                f"True pos: {true_pos}, True neg: {true_neg}, False pos: {false_pos}, False neg: {false_neg}, Total: {total}")
                        embeddings = model(
                            **val_data['contexts'])['last_hidden_state'][:, 0, :]
                        val_logits = classification_head(embeddings)
                        val_logits = val_logits.view(
                            -1, args.neg_samples_eval[1] + 1)
                        labels = torch.zeros_like(val_logits)
                        labels[:, 0] = 1
                        val_loss = loss_fn(val_logits, labels)

                        # gather the results from all processes
                        val_logits = accelerator.pad_across_processes(
                            val_logits, dim=0, pad_index=-1)
                        labels = accelerator.pad_across_processes(
                            labels, dim=0, pad_index=-1)
                        noise = accelerator.pad_across_processes(
                            val_data['noises'], dim=0, pad_index=-1)

                        val_logits = accelerator.gather_for_metrics(
                            val_logits).to('cpu')
                        labels = accelerator.gather_for_metrics(
                            labels).to('cpu')
                        noise = accelerator.gather_for_metrics(
                            noise).to('cpu')

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

                        # compute discritized scores for each noise type
                        for i in range(len(missing_map)):
                            noise_part = noise == i
                            labels_part = labels[noise_part]

                            for k in [1, 5, 10]:
                                noise_perf[i]['mrr'][str(k)] += torch.sum(
                                    1 / (torch.nonzero(labels_part[:, :k])[:, 1].float() + 1)).item()
                                noise_perf[i]['hits'][str(k)] += torch.sum(
                                    torch.sum(labels_part[:, :k], dim=1)).item()
                                noise_perf[i]['ndcg'][str(k)] += torch.sum(
                                    torch.sum(labels_part[:, :k] / torch.log2(torch.arange(2, k + 2).float()), dim=1)).item()
                            noise_perf[i]['mrr']['max'] += torch.sum(
                                1 / (torch.nonzero(labels_part)[:, 1].float() + 1)).item()
                            noise_perf[i]['hits']['max'] += torch.sum(
                                torch.sum(labels_part, dim=1)).item()
                            noise_perf[i]['ndcg']['max'] += torch.sum(
                                torch.sum(labels_part / torch.log2(torch.arange(2, labels_part.shape[1] + 2).float()), dim=1)).item()

                            noise_perf[i]['n_lists'] += len(labels_part)

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
                    for i in range(len(missing_map)):
                        if noise_perf[i]['n_lists'] > 0:
                            noise_perf[i]['mrr'] = {
                                k: v / noise_perf[i]['n_lists'] for k, v in noise_perf[i]['mrr'].items()}
                            noise_perf[i]['hits'] = {
                                k: v / noise_perf[i]['n_lists'] for k, v in noise_perf[i]['hits'].items()}
                            noise_perf[i]['ndcg'] = {
                                k: v / noise_perf[i]['n_lists'] for k, v in noise_perf[i]['ndcg'].items()}
                    running_val_loss /= len(val_loader)

                    for k in ['1', '5', '10', 'max']:
                        logger.info(f"MRR@{k}: {mrr_at_k[k]}")
                        logger.info(f"Hits@{k}: {hits_at_k[k]}")
                        logger.info(f"NDCG@{k}: {ndcg_at_k[k]}")
                    for i in range(len(missing_map)):
                        if noise_perf[i]['n_lists'] > 0:
                            logger.info(
                                f"Noise strategy {missing_map_rev[i]}:")
                            for k in ['1', '5', '10', 'max']:
                                logger.info(
                                    f"\t- MRR@{k}: {noise_perf[i]['mrr'][k]}")
                                logger.info(
                                    f"\t- Hits@{k}: {noise_perf[i]['hits'][k]}")
                                logger.info(
                                    f"\t- NDCG@{k}: {noise_perf[i]['ndcg'][k]}")
                    logger.info(f"Accuracy: {accuracy}")
                    logger.info(f"Precision: {precision}")
                    logger.info(f"Recall: {recall}")
                    logger.info(f"F1: {f1}")
                    logger.info(f"Validation loss: {running_val_loss}")

                    if accelerator.is_main_process:
                        for k in ['1', '5', '10', 'max']:
                            writer.add_scalar('val_stage2/mrr@' + k,
                                              mrr_at_k[k], step)
                            writer.add_scalar('val_stage2/hits@' + k,
                                              hits_at_k[k], step)
                            writer.add_scalar('val_stage2/ndcg@' + k,
                                              ndcg_at_k[k], step)
                        writer.add_scalar(
                            'val_stage2/accuracy', accuracy, step)
                        writer.add_scalar(
                            'val_stage2/precision', precision, step)
                        writer.add_scalar('val_stage2/recall', recall, step)
                        writer.add_scalar('val_stage2/f1', f1, step)
                        writer.add_scalar('val_stage2/loss',
                                          running_val_loss, step)
                        for i in range(len(noise_map)):
                            if noise_perf[i]['n_lists'] > 0:
                                for k in ['1', '5', '10', 'max']:
                                    writer.add_scalar(
                                        f'val_noise_stage2/{noise_map_rev[i]}_mrr@' + k, noise_perf[i]['mrr'][k], step)
                                    writer.add_scalar(
                                        f'val_noise_stage2/{noise_map_rev[i]}_hits@' + k, noise_perf[i]['hits'][k], step)
                                    writer.add_scalar(
                                        f'val_noise_stage2/{noise_map_rev[i]}_ndcg@' + k, noise_perf[i]['ndcg'][k], step)

                model.train()
                torch.cuda.empty_cache()
                gc.collect()

    logger.info("Training finished")
    if accelerator.is_main_process:
        writer.close()

    accelerator.wait_for_everyone()
    # save last version of models
    accelerator.unwrap_model(model).save_pretrained(os.path.join(
        output_dir, f"model"))
    torch.save(accelerator.unwrap_model(classification_head).state_dict(), os.path.join(
        output_dir, f"classification_head.pth"))
