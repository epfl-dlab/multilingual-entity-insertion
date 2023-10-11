import argparse
import logging
import os
import time

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

multiprocess.set_start_method("spawn", force=True)


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
                        help='Number of epochs to freeze all lazyers except classification head')
    parser.add_argument('--freeze_layers', type=int, default=2,
                        help='Number of initial layers to freeze')
    parser.add_argument('--head_lr_factor', type=float, default=1,
                        help='Factor for learning rate of classification head')
    parser.set_defaults(resume=False)

    args = parser.parse_args()

    # check if checkpoint_dir is provided and exists if resuming training
    if args.resume:
        if args.checkpoint_dir is None:
            raise ValueError(
                "Please provide checkpoint directory with --checkpoint_dir")
        if not os.path.exists(args.checkpoint_dir):
            raise ValueError(
                f"Checkpoint directory {args.checkpoint_dir} does not exist")

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
    if not os.path.exists('output'):
        os.makedirs('output', exist_ok=True)

    output_dir = os.path.join('output', date_time)
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
        sources, contexts, targets = [], [], []
        labels = []
        for item in input:
            source_input = f"{item['source_title']}{tokenizer.sep_token}{item['source_lead']}"
            context_input = f"{item['source_section']}{tokenizer.sep_token}{item['link_context']}"
            target_input = f"{item['target_title']}{tokenizer.sep_token}{item['target_lead']}"

            sources.append(source_input)
            contexts.append(context_input)
            targets.append(target_input)
            labels.append(item['label'])
        sources = tokenizer(sources, padding=True,
                            truncation=True, return_tensors='pt')
        contexts = tokenizer(contexts, padding=True,
                             truncation=True, return_tensors='pt')
        targets = tokenizer(targets, padding=True,
                            truncation=True, return_tensors='pt')
        return {'sources': sources, 'contexts': contexts, 'targets': targets, 'labels': torch.tensor(labels)}

    logger.info("Loading datasets")
    train_set = WikiDataset(args.data_dir, 'train')
    val_set = WikiDataset(args.data_dir, 'val')
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
            output_source = model(**data['sources'])
            output_context = model(**data['contexts'])
            output_target = model(**data['targets'])
            embeddings = [output_source['last_hidden_state'][:, 0, :],
                          output_context['last_hidden_state'][:, 0, :],
                          output_target['last_hidden_state'][:, 0, :]]
            embeddings = torch.cat(embeddings, dim=1)
            logits = classification_head(embeddings)
            loss = loss_fn(logits, data['labels']) / args.ga_steps
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
                    total = 0
                    running_val_loss = 0
                    for j, val_data in (pbar := tqdm(enumerate(val_loader), total=len(val_loader))):
                        if j % 250 == 0:
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
                        val_loss = loss_fn(val_logits, val_data['labels'])

                        # gather the results from all processes
                        val_logits = accelerator.pad_across_processes(
                            val_logits, dim=0, pad_index=-1)
                        labels = accelerator.pad_across_processes(
                            val_data['labels'], dim=0, pad_index=-1)
                        val_logits = accelerator.gather_for_metrics(
                            val_logits).to('cpu')
                        labels = accelerator.gather_for_metrics(
                            labels).to('cpu')

                        val_loss = accelerator.gather_for_metrics(
                            val_loss).to('cpu')
                        running_val_loss += val_loss.mean().item()

                        # measure true positives, true negatives, false positives, false negatives
                        # use softmax to get probabilities
                        probs = torch.softmax(val_logits, dim=1)
                        preds = torch.argmax(probs, dim=1)

                        true_pos += torch.sum((preds == 1)
                                              & (labels == 1)).item()
                        true_neg += torch.sum((preds == 0)
                                              & (labels == 0)).item()
                        false_pos += torch.sum((preds == 1)
                                               & (labels == 0)).item()
                        false_neg += torch.sum((preds == 0)
                                               & (labels == 1)).item()
                        total += len(labels)
                        
                        if j == len(val_loader) - 1:
                            pbar.set_description(
                                f"True pos: {true_pos}, True neg: {true_neg}, False pos: {false_pos}, False neg: {false_neg}, Total: {total}")
                    # calculate accuracy, precision, recall, f1 score
                    accuracy = (true_pos + true_neg) / total
                    precision = true_pos / \
                        (true_pos + false_pos) if true_pos + false_pos > 0 else 0
                    recall = true_pos / \
                        (true_pos + false_neg) if true_pos + false_neg > 0 else 0
                    f1 = 2 * precision * recall / \
                        (precision + recall) if precision + recall > 0 else 0
                    running_val_loss /= len(val_loader)
                    logger.info(f"Accuracy: {accuracy}")
                    logger.info(f"Precision: {precision}")
                    logger.info(f"Recall: {recall}")
                    logger.info(f"F1: {f1}")
                    logger.info(f"Validation loss: {running_val_loss}")
                    logger.info(f"Model distance: {model_distance}")
                    if accelerator.is_main_process:
                        writer.add_scalar('val/accuracy', accuracy, step)
                        writer.add_scalar('val/precision', precision, step)
                        writer.add_scalar('val/recall', recall, step)
                        writer.add_scalar('val/f1', f1, step)
                        writer.add_scalar('val/loss', running_val_loss, step)
                        writer.add_scalar('model/distance',
                                          model_distance, step)
                model.train()

        scheduler.step()

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
