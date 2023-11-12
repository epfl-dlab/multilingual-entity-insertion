import argparse
import logging
from operator import is_
import os
import time
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
from urllib.parse import unquote
from nltk import sent_tokenize

multiprocess.set_start_method("spawn", force=True)


def mask_negative_contexts(context, probs):
    sentences = [sentence.strip()
                 for sentence in sent_tokenize(context) if sentence.strip()]
    if len(sentences) <= 2:
        probs['mask_span'] = 0
    if len(sentences) == 1:
        probs['mask_sentence'] = 0
    words = []
    for sentence in sentences:
        words.extend([word for word in sentence.replace(
            '\n', ' ').split() if word.strip()])
    if len(words) == 1:
        probs['mask_word'] = 0

    if probs['mask_span'] + probs['mask_sentence'] + probs['mask_word'] + probs['no_mask'] == 0:
        probs['no_mask'] = 1

    mask_strategy = random.choices(['mask_span', 'mask_sentence', 'mask_mention', 'no_mask'],
                                   weights=[probs['mask_span'], probs['mask_sentence'],
                                            probs['mask_mention'], probs['no_mask']],
                                   k=1)[0]
    if mask_strategy == 'no_mask':
        return context
    if mask_strategy == 'mask_mention':
        sentence_index = random.randint(0, len(sentences) - 1)
        words = sentences[sentence_index].split(' ')
        mask_index = random.randint(0, len(words) - 1)
        masked_context = ''
        for i, sentence in sentences:
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
        mask_length = random.randint(2, len(sentences) - 1)
        start_index = random.randint(0, len(sentences) - mask_length)
        return " ".join(sentences[:start_index]) + " " + " ".join(sentences[start_index + mask_length:])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_name', type=str,
                        default='bert-base-uncased', help='Model name or path to model')
    parser.add_argument('--data_dir', type=str,
                        required=True, help='Data directory')
    parser.add_argument('--num_epochs', type=int,
                        default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of workers for dataloader')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
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
    parser.add_argument('--full_freeze_epochs', type=int, default=0,
                        help='Number of epochs to freeze all layers except classification head')
    parser.add_argument('--freeze_layers', type=int, default=2,
                        help='Number of initial layers to freeze')
    parser.add_argument('--head_lr_factor', type=float, default=1,
                        help='Factor for learning rate of classification head')
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
    parser.add_argument('--neg_samples_train', type=int, default=1,
                        help='Number of negative samples for training')
    parser.add_argument('--neg_samples_eval', type=int, default=20,
                        help='Number of negative samples for evaluation')
    parser.add_argument('--temperature', type=float, default=1,
                        help='Temperature for softmax')
    parser.add_argument('--insert_mentions', type=str, choices=[
                        'none', 'target', 'candidates'], default='none', help='Where to insert mention knowledge')
    parser.add_argument('--insert_section', action='store_true',
                        help='Whether to insert section knowledge')
    parser.add_argument('--mask_negatives', action='store_true',
                        help='Whether to apply masking to negative samples')
    parser.set_defaults(resume=False, insert_section=False,
                        mask_negatives=False)

    args = parser.parse_args()

    # check if checkpoint_dir is provided and exists if resuming training
    if args.resume:
        if args.checkpoint_dir is None:
            raise ValueError(
                "Please provide checkpoint directory with --checkpoint_dir")
        if not os.path.exists(args.checkpoint_dir):
            raise ValueError(
                f"Checkpoint directory {args.checkpoint_dir} does not exist")

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
    date_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    tb_dir = os.path.join('runs', date_time)
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir, exist_ok=True)
    if accelerator.is_main_process:
        writer = SummaryWriter(tb_dir)

    # create directory for logs and checkpoints
    if not os.path.exists('output_indep'):
        os.makedirs('output_indep', exist_ok=True)

    output_dir = os.path.join('output_indep', date_time)
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
            model = AutoModel.from_pretrained(args.checkpoint_dir)
            logger.info("Model loaded from checkpoint directory")
        except OSError:
            logger.info("Could not load model from checkpoint directory")
            logger.info("Initializing model from provided model name")
            model = AutoModel.from_pretrained(args.model_name)
        try:
            classification_head = Sequential(nn.Linear(model.config.hidden_size * 3, model.config.hidden_size),
                                             nn.ReLU(),
                                             nn.Linear(model.config.hidden_size, 2))
            classification_head.load_state_dict(torch.load(os.path.join(
                args.checkpoint_dir, 'classification_head.pth'), map_location='cpu'))
            logger.info("Classification head loaded from checkpoint directory")
        except OSError:
            logger.info(
                "Could not load classification head from checkpoint directory")
            logger.info("Initializing classification head with random weights")
            classification_head = Sequential(nn.Linear(model.config.hidden_size * 3, model.config.hidden_size),
                                             nn.ReLU(),
                                             nn.Linear(model.config.hidden_size, 2))
    else:
        logger.info("Initializing model")
        model = AutoModel.from_pretrained(args.model_name)
        classification_head = Sequential(nn.Linear(model.config.hidden_size * 3, model.config.hidden_size),
                                         nn.ReLU(),
                                         nn.Linear(model.config.hidden_size, 2))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.save_pretrained(os.path.join(output_dir, 'tokenizer'))

    # store model weights to keep track of model distance
    model_weights = torch.cat([param.data.flatten()
                              for param in model.parameters()]).to('cpu')

    logger.info("Initializing optimizer")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # add classification head to optimizer
    optimizer.add_param_group(
        {'params': classification_head.parameters(), 'lr': args.lr * args.head_lr_factor})

    # set-up scheduler
    scheduler = ExponentialLR(optimizer, gamma=args.gamma_lr)

    # define loss
    loss_fn = nn.CrossEntropyLoss()

    def collator(input):
        output = {'sources': [], 'contexts': [], 'targets': [], 'noises': []}
        if input[0]['split'] == 'train':
            for i in range(args.neg_samples_train):
                output[f"contexts_neg_{i}"] = []
                output[f"strategy_neg_{i}"] = []
            for item in input:
                found = False
                if (item['link_context'][:item['context_span_start_index']] + item['link_context'][item['context_span_end_index']:]).strip() != '':
                    if item['context_span_start_index'] <= item['context_sentence_start_index'] and item['context_span_end_index'] >= item['context_sentence_end_index']:
                        noise_types = [
                            'mask_span', 'mask_sentence', 'mask_mention', 'no_mask']
                        found = True
                if not found and (item['link_context'][:item['context_sentence_start_index']] + item['link_context'][item['context_sentence_end_index']:]).strip() != '':
                    if item['context_sentence_start_index'] <= item['context_mention_start_index'] and item['context_sentence_end_index'] > item['context_mention_end_index'] + 1:
                        noise_types = ['mask_sentence',
                                       'mask_mention', 'no_mask']
                        found = True
                if not found and (item['link_context'][:item['context_mention_start_index']] + item['link_context'][item['context_mention_end_index']:]).strip() != '':
                    noise_types = ['mask_mention', 'no_mask']
                    found = True
                if not found:
                    noise_types = ['no_mask']

                noise_type = random.choices(
                    noise_types, weights=weights[-len(noise_types):], k=1)[0]
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

                source_input = f"{item['source_title']}{tokenizer.sep_token}{item['source_lead']}"
                if args.insert_section:
                    context_input = f"{item['source_section']}{tokenizer.sep_token}"
                else:
                    context_input = ''
                if args.insert_mentions == 'candidates':
                    context_input += f"{mention_map[item['target_title']]}{tokenizer.sep_token}{item['link_context']}"
                else:
                    context_input += f"{item['link_context']}"

                if args.insert_mentions == 'target':
                    target_input = f"{item['target_title']}{tokenizer.sep_token}{mention_map[item['target_title']]}{tokenizer.sep_token}{item['target_lead']}"
                else:
                    target_input = f"{item['target_title']}{tokenizer.sep_token}{item['target_lead']}"

                output['noises'].append(noise_map[noise_type])
                output['sources'].append(source_input)
                output['contexts'].append(context_input)
                output['targets'].append(target_input)

                mask_probs = {'no_mask': args.no_mask_perc, 'mask_mention': args.mask_mention_perc,
                              'mask_sentence': args.mask_sentence_perc, 'mask_span': args.mask_span_perc}
                for i in range(args.neg_samples_train):
                    source_section_neg = item[f"source_section_neg_{i}"]
                    link_context_neg = item[f"link_context_neg_{i}"]
                    if args.mask_negatives:
                        link_context_neg = mask_negative_contexts(
                            link_context_neg, mask_probs)

                    if args.insert_section:
                        context_input = f"{source_section_neg}{tokenizer.sep_token}"
                    else:
                        context_input = ''

                    if args.insert_mentions == 'candidates':
                        context_input += f"{mention_map[item['target_title']]}{tokenizer.sep_token}{link_context_neg}"
                    else:
                        context_input += f"{link_context_neg}"
                    output[f"contexts_neg_{i}"].append(context_input)
                    output[f"strategy_neg_{i}"].append(
                        neg_map[item[f'neg_type_neg_{i}']])
        else:
            for i in range(args.neg_samples_eval):
                output[f"contexts_neg_{i}"] = []
                output[f"strategy_neg_{i}"] = []
            for item in input:
                source_input = f"{item['source_title']}{tokenizer.sep_token}{item['source_lead']}"
                if args.insert_section:
                    context_input = f"{item['source_section']}{tokenizer.sep_token}"
                else:
                    context_input = ''
                if args.insert_mentions == 'candidates':
                    context_input += f"{mention_map[item['target_title']]}{tokenizer.sep_token}{item['link_context']}"
                else:
                    context_input += f"{item['link_context']}"

                if args.insert_mentions == 'target':
                    target_input = f"{item['target_title']}{tokenizer.sep_token}{mention_map[item['target_title']]}{tokenizer.sep_token}{item['target_lead']}"
                else:
                    target_input = f"{item['target_title']}{tokenizer.sep_token}{item['target_lead']}"

                output['noises'].append(noise_map[item['noise_strategy']])
                output['sources'].append(source_input)
                output['contexts'].append(context_input)
                output['targets'].append(target_input)
                mask_probs = {'no_mask': args.no_mask_perc, 'mask_mention': args.mask_mention_perc,
                              'mask_sentence': args.mask_sentence_perc, 'mask_span': args.mask_span_perc}
                for i in range(args.neg_samples_eval):
                    source_section_neg = item[f"source_section_neg_{i}"]
                    link_context_neg = item[f"link_context_neg_{i}"]
                    if args.mask_negatives:
                        link_context_neg = mask_negative_contexts(
                            link_context_neg, mask_probs)

                    if args.insert_section:
                        context_input = f"{source_section_neg}{tokenizer.sep_token}"
                    else:
                        context_input = ''

                    if args.insert_mentions == 'candidates':
                        context_input += f"{mention_map[item['target_title']]}{tokenizer.sep_token}{link_context_neg}"
                    else:
                        context_input += f"{link_context_neg}"
                    output[f"contexts_neg_{i}"].append(context_input)
                    output[f"strategy_neg_{i}"].append(
                        neg_map[item[f'neg_type_neg_{i}']])

        output['sources'] = tokenizer(
            output['sources'], padding=True, truncation=True, return_tensors='pt', max_length=args.max_tokens)
        output['targets'] = tokenizer(
            output['targets'], padding=True, truncation=True, return_tensors='pt', max_length=args.max_tokens)
        output['contexts'] = tokenizer(
            output['contexts'], padding=True, truncation=True, return_tensors='pt', max_length=args.max_tokens)
        output['noises'] = torch.tensor(output['noises'])
        if input[0]['split'] == 'train':
            for i in range(args.neg_samples_train):
                output[f"contexts_neg_{i}"] = tokenizer(
                    output[f"contexts_neg_{i}"], padding=True, truncation=True, return_tensors='pt', max_length=args.max_tokens)
                output[f"strategy_neg_{i}"] = torch.tensor(
                    output[f"strategy_neg_{i}"])
        else:
            for i in range(args.neg_samples_eval):
                output[f"contexts_neg_{i}"] = tokenizer(
                    output[f"contexts_neg_{i}"], padding=True, truncation=True, return_tensors='pt', max_length=args.max_tokens)
                output[f"strategy_neg_{i}"] = torch.tensor(
                    output[f"strategy_neg_{i}"])
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

    logger.info("Loading mention knowledge")
    mentions = pd.read_parquet(os.path.join(
        args.data_dir, 'mentions.parquet')).to_dict('records')
    mention_map = {}
    for mention in mentions:
        target_title = unquote(mention['target_title']).replace('_', ' ')
        if target_title in mention_map:
            mention_map[target_title] += ' ' + mention['mention']
        else:
            mention_map[target_title] = mention['mention']

    if args.full_freeze_epochs > 0:
        logger.info(
            f"Freezing all layers except classification head for {args.full_freeze_epochs} epochs")
        for param in model.parameters():
            param.requires_grad = False
        for param in classification_head.parameters():
            param.requires_grad = True
    else:
        logger.info(f"Freezing first {args.freeze_layers} layers")
        for param in model.base_model.embeddings.parameters():
            param.requires_grad = False
        for param in model.base_model.encoder.layer[:args.freeze_layers].parameters():
            param.requires_grad = False

    # prepare all objects with accelerator
    model, classification_head, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, classification_head, optimizer, train_loader, val_loader, scheduler)

    logger.info("Starting training")
    step = 0
    running_loss = 0
    for epoch in range(args.num_epochs):
        for index, data in enumerate(train_loader):
            step += 1
            # multiple forward passes accumulate gradients
            # source: https://discuss.pytorch.org/t/multiple-model-forward-followed-by-one-loss-backward/20868
            output_source = model(**data['sources'])
            output_context_pos = model(**data['contexts'])
            output_target = model(**data['targets'])
            embeddings_pos = [output_source['last_hidden_state'][:, 0, :],
                              output_context_pos['last_hidden_state'][:, 0, :],
                              output_target['last_hidden_state'][:, 0, :]]
            embeddings_pos = torch.cat(embeddings_pos, dim=1)
            logits = classification_head(embeddings_pos)
            loss = loss_fn(logits / args.temperature, torch.ones(logits.shape[0]).long().to(
                device)) / args.ga_steps / (args.neg_samples_train + 1)
            for i in range(args.neg_samples_train):
                output_context_neg = model(**data[f"contexts_neg_{i}"])
                embeddings_neg = [output_source['last_hidden_state'][:, 0, :],
                                  output_context_neg['last_hidden_state'][:, 0, :],
                                  output_target['last_hidden_state'][:, 0, :]]
                embeddings_neg = torch.cat(embeddings_neg, dim=1)
                logits = classification_head(embeddings_neg)
                loss += loss_fn(logits / args.temperature, torch.zeros(logits.shape[0]).long().to(
                    device)) / args.ga_steps / (args.neg_samples_train + 1)
            accelerator.backward(loss)
            if (index + 1) % args.ga_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            # save running loss
            running_loss += loss.item() * args.ga_steps
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
                    f"Encoder learning rate: {scheduler.get_last_lr()[0]}")
                logger.info(
                    f"Classification head learning rate: {scheduler.get_last_lr()[1]}")

            # save model
            if step % args.save_steps == 0:
                logger.info(f"Step {step}: saving model")
                accelerator.wait_for_everyone()
                # accelerator needs to unwrap model and classification head
                accelerator.unwrap_model(model).save_pretrained(os.path.join(
                    output_dir, f"model_{step}"))
                torch.save(accelerator.unwrap_model(classification_head).state_dict(), os.path.join(
                    output_dir, f"classification_head_{step}.pth"))

            # evaluate model
            if step % args.eval_steps == 0:
                logger.info(f"Step {step}: evaluating model")
                model.eval()
                with torch.no_grad():
                    # compare current model weights to initial model weights
                    current_model_weights = torch.cat(
                        [param.data.flatten() for param in model.parameters()]).to('cpu')
                    model_distance = torch.norm(
                        current_model_weights - model_weights) / torch.norm(model_weights)

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
                        output_source = model(**val_data['sources'])
                        output_context = model(**val_data['contexts'])
                        output_target = model(**val_data['targets'])
                        val_embeddings = [output_source['last_hidden_state'][:, 0, :],
                                          output_context['last_hidden_state'][:, 0, :],
                                          output_target['last_hidden_state'][:, 0, :]]
                        val_embeddings = torch.cat(val_embeddings, dim=1)
                        val_logits = classification_head(val_embeddings)
                        val_loss = loss_fn(val_logits / args.temperature, torch.ones(
                            val_logits.shape[0]).long().to(device)) / (args.neg_samples_eval + 1)

                        val_logits_negs = []
                        neg_strategies = []
                        for i in range(args.neg_samples_eval):
                            neg_strategies.append(
                                val_data[f"strategy_neg_{i}"])
                            output_context_neg = model(
                                **val_data[f"contexts_neg_{i}"])
                            val_embeddings_neg = [output_source['last_hidden_state'][:, 0, :],
                                                  output_context_neg['last_hidden_state'][:, 0, :],
                                                  output_target['last_hidden_state'][:, 0, :]]
                            val_embeddings_neg = torch.cat(
                                val_embeddings_neg, dim=1)
                            val_logits_neg = classification_head(
                                val_embeddings_neg)
                            val_logits_negs.append(val_logits_neg)
                            val_loss += loss_fn(val_logits_neg / args.temperature, torch.zeros(
                                val_logits_neg.shape[0]).long().to(device)) / (args.neg_samples_eval + 1)

                        # gather the results from all processes
                        val_logits = accelerator.pad_across_processes(
                            val_logits, dim=0, pad_index=-1)
                        for i in range(len(val_logits_negs)):
                            val_logits_negs[i] = accelerator.pad_across_processes(
                                val_logits_negs[i], dim=0, pad_index=-1)
                        for i in range(len(neg_strategies)):
                            neg_strategies[i] = accelerator.pad_across_processes(
                                neg_strategies[i], dim=0, pad_index=-1)
                        noise = accelerator.pad_across_processes(
                            val_data['noises'], dim=0, pad_index=-1)

                        val_logits = accelerator.gather_for_metrics(
                            val_logits).to('cpu')
                        for i in range(len(val_logits_negs)):
                            val_logits_negs[i] = accelerator.gather_for_metrics(
                                val_logits_negs[i]).to('cpu')
                        for i in range(len(neg_strategies)):
                            neg_strategies[i] = accelerator.gather_for_metrics(
                                neg_strategies[i]).to('cpu')
                        noise = accelerator.gather_for_metrics(
                            noise).to('cpu')

                        val_loss = accelerator.gather_for_metrics(
                            val_loss).to('cpu')
                        running_val_loss += val_loss.mean().item()

                        n_lists += len(val_logits)

                        # join positive and negative logits into one tensor containing the probability of label=1
                        # shape (batch_size, (neg_samples + 1), 2)
                        val_logits = torch.stack(
                            [val_logits] + val_logits_negs, dim=1)
                        # apply softmax to get the probability of label=1
                        # shape (batch_size, (neg_samples + 1))
                        probs = torch.softmax(val_logits, dim=2)[:, :, 1]
                        # get the labels
                        # shape (batch_size, (neg_samples + 1))
                        labels = torch.zeros_like(probs)
                        labels[:, 0] = 1

                        # calculate mrr, hits@k, ndcg@k
                        # sort probabilities in descending order and labels accordingly
                        # shape (batch_size, (neg_samples + 1))
                        probs, indices = torch.sort(
                            probs, dim=1, descending=True)
                        labels = torch.gather(labels, dim=1, index=indices)
                        # calculate mrr
                        # shape ()
                        mrr_at_k['1'] += torch.sum(
                            1 / (torch.nonzero(labels[:, :1])[:, 1].float() + 1)).item()
                        mrr_at_k['5'] += torch.sum(
                            1 / (torch.nonzero(labels[:, :5])[:, 1].float() + 1)).item()
                        mrr_at_k['10'] += torch.sum(
                            1 / (torch.nonzero(labels[:, :10])[:, 1].float() + 1)).item()
                        mrr_at_k['max'] += torch.sum(
                            1 / (torch.nonzero(labels)[:, 1].float() + 1)).item()

                        # calculate hits@k
                        # shape ()
                        hits_at_k['1'] += torch.sum(
                            torch.sum(labels[:, :1], dim=1)).item()
                        hits_at_k['5'] += torch.sum(
                            torch.sum(labels[:, :5], dim=1)).item()
                        hits_at_k['10'] += torch.sum(
                            torch.sum(labels[:, :10], dim=1)).item()
                        hits_at_k['max'] += torch.sum(
                            torch.sum(labels, dim=1)).item()

                        # calculate ndcg@k
                        # shape ()
                        ndcg_at_k['1'] += torch.sum(
                            torch.sum(labels[:, :1] / torch.log2(torch.arange(2, 3).float()), dim=1)).item()
                        ndcg_at_k['5'] += torch.sum(
                            torch.sum(labels[:, :5] / torch.log2(torch.arange(2, 7).float()), dim=1)).item()
                        ndcg_at_k['10'] += torch.sum(
                            torch.sum(labels[:, :10] / torch.log2(torch.arange(2, 12).float()), dim=1)).item()
                        ndcg_at_k['max'] += torch.sum(
                            torch.sum(labels / torch.log2(torch.arange(2, labels.shape[1] + 2).float()), dim=1)).item()

                        # compute discritized scores for each noise type
                        for i in range(len(noise_map)):
                            noise_part = noise == i
                            labels_part = labels[noise_part]

                            noise_perf[i]['mrr']['1'] += torch.sum(
                                1 / (torch.nonzero(labels_part[:, :1])[:, 1].float() + 1)).item()
                            noise_perf[i]['mrr']['5'] += torch.sum(
                                1 / (torch.nonzero(labels_part[:, :5])[:, 1].float() + 1)).item()
                            noise_perf[i]['mrr']['10'] += torch.sum(
                                1 / (torch.nonzero(labels_part[:, :10])[:, 1].float() + 1)).item()
                            noise_perf[i]['mrr']['max'] += torch.sum(
                                1 / (torch.nonzero(labels_part)[:, 1].float() + 1)).item()

                            noise_perf[i]['hits']['1'] += torch.sum(
                                torch.sum(labels_part[:, :1], dim=1)).item()
                            noise_perf[i]['hits']['5'] += torch.sum(
                                torch.sum(labels_part[:, :5], dim=1)).item()
                            noise_perf[i]['hits']['10'] += torch.sum(
                                torch.sum(labels_part[:, :10], dim=1)).item()
                            noise_perf[i]['hits']['max'] += torch.sum(
                                torch.sum(labels_part, dim=1)).item()

                            noise_perf[i]['ndcg']['1'] += torch.sum(
                                torch.sum(labels_part[:, :1] / torch.log2(torch.arange(2, 3).float()), dim=1)).item()
                            noise_perf[i]['ndcg']['5'] += torch.sum(
                                torch.sum(labels_part[:, :5] / torch.log2(torch.arange(2, 7).float()), dim=1)).item()
                            noise_perf[i]['ndcg']['10'] += torch.sum(
                                torch.sum(labels_part[:, :10] / torch.log2(torch.arange(2, 12).float()), dim=1)).item()
                            noise_perf[i]['ndcg']['max'] += torch.sum(
                                torch.sum(labels_part / torch.log2(torch.arange(2, labels_part.shape[1] + 2).float()), dim=1)).item()

                            noise_perf[i]['n_lists'] += len(labels_part)

                        preds = (probs > 0.5).long()
                        true_pos += torch.sum((preds == 1)
                                              & (labels == 1)).item()
                        true_neg += torch.sum((preds == 0)
                                              & (labels == 0)).item()
                        false_pos += torch.sum((preds == 1)
                                               & (labels == 0)).item()
                        false_neg += torch.sum((preds == 0)
                                               & (labels == 1)).item()

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

                    logger.info(f"MRR@1: {mrr_at_k['1']}")
                    logger.info(f"MRR@5: {mrr_at_k['5']}")
                    logger.info(f"MRR@10: {mrr_at_k['10']}")
                    logger.info(f"MRR@max: {mrr_at_k['max']}")
                    logger.info(f"Hits@1: {hits_at_k['1']}")
                    logger.info(f"Hits@5: {hits_at_k['5']}")
                    logger.info(f"Hits@10: {hits_at_k['10']}")
                    logger.info(f"Hits@max: {hits_at_k['max']}")
                    logger.info(f"NDCG@1: {ndcg_at_k['1']}")
                    logger.info(f"NDCG@5: {ndcg_at_k['5']}")
                    logger.info(f"NDCG@10: {ndcg_at_k['10']}")
                    logger.info(f"NDCG@max: {ndcg_at_k['max']}")
                    for i in range(len(noise_map)):
                        if noise_perf[i]['n_lists'] > 0:
                            logger.info(f"Noise strategy {noise_map_rev[i]}:")
                            logger.info(
                                f"\t- MRR@1: {noise_perf[i]['mrr']['1']}")
                            logger.info(
                                f"\t- MRR@5: {noise_perf[i]['mrr']['5']}")
                            logger.info(
                                f"\t- MRR@10: {noise_perf[i]['mrr']['10']}")
                            logger.info(
                                f"\t- MRR@max: {noise_perf[i]['mrr']['max']}")
                            logger.info(
                                f"\t- Hits@1: {noise_perf[i]['hits']['1']}")
                            logger.info(
                                f"\t- Hits@5: {noise_perf[i]['hits']['5']}")
                            logger.info(
                                f"\t- Hits@10: {noise_perf[i]['hits']['10']}")
                            logger.info(
                                f"\t- Hits@max: {noise_perf[i]['hits']['max']}")
                            logger.info(
                                f"\t- NDCG@1: {noise_perf[i]['ndcg']['1']}")
                            logger.info(
                                f"\t- NDCG@5: {noise_perf[i]['ndcg']['5']}")
                            logger.info(
                                f"\t- NDCG@10: {noise_perf[i]['ndcg']['10']}")
                            logger.info(
                                f"\t- NDCG@max: {noise_perf[i]['ndcg']['max']}")
                    logger.info(f"Accuracy: {accuracy}")
                    logger.info(f"Precision: {precision}")
                    logger.info(f"Recall: {recall}")
                    logger.info(f"F1: {f1}")
                    logger.info(f"Validation loss: {running_val_loss}")
                    logger.info(f"Model distance: {model_distance}")

                    if accelerator.is_main_process:
                        writer.add_scalar('val/mrr@1', mrr_at_k['1'], step)
                        writer.add_scalar('val/mrr@5', mrr_at_k['5'], step)
                        writer.add_scalar('val/mrr@10', mrr_at_k['10'], step)
                        writer.add_scalar('val/mrr@max', mrr_at_k['max'], step)
                        writer.add_scalar('val/hits@1', hits_at_k['1'], step)
                        writer.add_scalar('val/hits@5', hits_at_k['5'], step)
                        writer.add_scalar('val/hits@10', hits_at_k['10'], step)
                        writer.add_scalar(
                            'val/hits@max', hits_at_k['max'], step)
                        writer.add_scalar('val/ndcg@1', ndcg_at_k['1'], step)
                        writer.add_scalar('val/ndcg@5', ndcg_at_k['5'], step)
                        writer.add_scalar('val/ndcg@10', ndcg_at_k['10'], step)
                        writer.add_scalar(
                            'val/ndcg@max', ndcg_at_k['max'], step)
                        writer.add_scalar('val/accuracy', accuracy, step)
                        writer.add_scalar('val/precision', precision, step)
                        writer.add_scalar('val/recall', recall, step)
                        writer.add_scalar('val/f1', f1, step)
                        writer.add_scalar('val/loss', running_val_loss, step)
                        writer.add_scalar('model/distance',
                                          model_distance, step)
                        for i in range(len(noise_map)):
                            if noise_perf[i]['n_lists'] > 0:
                                writer.add_scalar(
                                    f'val_noise/{noise_map_rev[i]}_mrr@1', noise_perf[i]['mrr']['1'], step)
                                writer.add_scalar(
                                    f'val_noise/{noise_map_rev[i]}_mrr@5', noise_perf[i]['mrr']['5'], step)
                                writer.add_scalar(
                                    f'val_noise/{noise_map_rev[i]}_mrr@10', noise_perf[i]['mrr']['10'], step)
                                writer.add_scalar(
                                    f'val_noise/{noise_map_rev[i]}_mrr@max', noise_perf[i]['mrr']['max'], step)
                                writer.add_scalar(
                                    f'val_noise/{noise_map_rev[i]}_hits@1', noise_perf[i]['hits']['1'], step)
                                writer.add_scalar(
                                    f'val_noise/{noise_map_rev[i]}_hits@5', noise_perf[i]['hits']['5'], step)
                                writer.add_scalar(
                                    f'val_noise/{noise_map_rev[i]}_hits@10', noise_perf[i]['hits']['10'], step)
                                writer.add_scalar(
                                    f'val_noise/{noise_map_rev[i]}_hits@max', noise_perf[i]['hits']['max'], step)
                                writer.add_scalar(
                                    f'val_noise/{noise_map_rev[i]}_ndcg@1', noise_perf[i]['ndcg']['1'], step)
                                writer.add_scalar(
                                    f'val_noise/{noise_map_rev[i]}_ndcg@5', noise_perf[i]['ndcg']['5'], step)
                                writer.add_scalar(
                                    f'val_noise/{noise_map_rev[i]}_ndcg@10', noise_perf[i]['ndcg']['10'], step)
                                writer.add_scalar(
                                    f'val_noise/{noise_map_rev[i]}_ndcg@max', noise_perf[i]['ndcg']['max'], step)

                model.train()
                torch.cuda.empty_cache()
                gc.collect()

        # unfreeze model if necessary
        if epoch + 1 == args.full_freeze_epochs:
            model = accelerator.unwrap_model(model)
            logger.info(
                f"Unfreezing model except first {args.freeze_layers} layers")
            for param in model.parameters():
                param.requires_grad = True
            for param in model.base_model.embeddings.parameters():
                param.requires_grad = False
            for param in model.base_model.encoder.layer[:args.freeze_layers].parameters():
                param.requires_grad = False
            model = accelerator.prepare(model)

    # close logger
    logger.info("Training finished")
    if accelerator.is_main_process:
        writer.close()
