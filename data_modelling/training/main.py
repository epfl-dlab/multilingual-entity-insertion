import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from accelerate.logging import get_logger
import logging
from accelerate import Accelerator
from data import WikiDataset
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from torch.utils.tensorboard import SummaryWriter


def collator(input):
    sentences_1, sentences_2, sentences_3 = [], [], []
    labels = []
    for item in input:
        sentences_1.append(item['sentence_1'])
        sentences_2.append(item['sentence_2'])
        sentences_3.append(item['sentence_3'])
        labels.append(item['label'])
    sentences_1 = tokenizer(sentences_1, padding=True,
                            truncation=True, return_tensors='pt')
    sentences_2 = tokenizer(sentences_2, padding=True,
                            truncation=True, return_tensors='pt')
    sentences_3 = tokenizer(sentences_3, padding=True,
                            truncation=True, return_tensors='pt')
    return {'inputs': [sentences_1, sentences_2, sentences_3], 'labels': torch.tensor(labels)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default='bert-base-uncased', help='Model name or path to model')
    parser.add_argument('--data_dir', type=str,
                        required=True, help='Data directory')
    parser.add_argument('--num_epochs', type=int,
                        default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for dataloader')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--gamma_lr', type=float,
                        default=0.9, help='Gamma for lr scheduler')
    parser.add_argument('--print_step', type=int, default=1000,
                        help='Number of steps between printing loss')
    parser.add_argument('--save_step', type=int, default=5000,
                        help='Number of steps between saving model')
    parser.add_argument('--eval_step', type=int, default=5000,
                        help='Number of steps between evaluating model on validation set')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint (needs --checkpoint_dir)')
    parser.add_argument('--checkpoint_dir', type=str,
                        help='Directory with checkpoint to resume training from')
    parser.add_argument('--ga_steps', type=int, default=1, help='Number of steps for gradient accumulation')
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
    accelerator = Accelerator(gradient_accumulation_steps=args.ga_steps)
    device = accelerator.device

    # set-up tensorboard
    if not os.path.exists('runs'):
        os.makedirs('runs')
    date_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    tb_dir = os.path.join('runs', date_time)
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)
    writer = SummaryWriter(tb_dir)

    # create directory for logs and checkpoints
    if not os.path.exists('output'):
        os.makedirs('output')

    output_dir = os.path.join('output', date_time)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # create logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(os.path.join(output_dir, 'log.txt')),
                                  logging.StreamHandler()])
    logger = get_logger(__name__, log_level="INFO")

    # log arguments
    logger.info("Arguments:")
    for arg in vars(args):
        logger.info(f"\t- {arg}: {getattr(args, arg)}")

    logger.info("Loading datasets")
    train_set = WikiDataset('train')
    val_set = WikiDataset('val')

    logger.info("Creating dataloaders")
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              drop_last=True,
                              collate_fn=collator)
    val_loader = DataLoader(val_set,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            drop_last=True,
                            collate_fn=collator)

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
            classification_head = torch.nn.Linear(
                model.config.hidden_size * 3, 2)
            classification_head.load_state_dict(torch.load(os.path.join(
                args.checkpoint_dir, 'classification_head.pth'), map_location='cpu'))
            logger.info("Classification head loaded from checkpoint directory")
        except OSError:
            logger.info(
                "Could not load classification head from checkpoint directory")
            logger.info("Initializing classification head with random weights")
            classification_head = torch.nn.Linear(
                model.config.hidden_size * 3, 2)
    else:
        logger.info("Initializing model")
        model = AutoModel.from_pretrained(args.model_name)
        classification_head = torch.nn.Linear(model.config.hidden_size * 3, 2)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.save_pretrained(os.path.join(output_dir, 'tokenizer'))
        model.to(device)

    logger.info("Initializing optimizer")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=args.gamma_lr)

    # prepare all objects with accelerator
    model, classification_head, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, classification_head, optimizer, train_loader, val_loader, scheduler)

    logger.info("Starting training")
    step = 0
    running_loss = 0
    for epoch in range(args.num_epochs):
        for data in train_loader:
            with accelerator.accumulate(model):
                step += 1
                embeddings = []
                for input in data['inputs']:
                    output = model(input['input_ids'], input['attention_mask'])
                    embeddings.append(output['last_hidden_state'][:, 0, :])
                embeddings = torch.cat(embeddings, dim=1)
                logits = classification_head(embeddings)
                loss = F.cross_entropy(logits, data['labels'])
                # save running loss
                running_loss += loss.item()
                # print loss
                if step % args.print_step == 0:
                    logger.info(
                        f"Step {step}: loss = {running_loss / args.print_step}")
                    writer.add_scalar('train/loss', running_loss / args.print_step, step)
                    running_loss = 0
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            # save model
            if step % args.save_step == 0:
                logger.info(f"Saving model at step {step}")
                accelerator.wait_for_everyone()
                # accelerator needs to unwrap model and classification head
                accelerator.unwrap_model(model).save_pretrained(os.path.join(
                    output_dir, f"model_{step}"))
                accelerator.unwrap_model(classification_head).save_pretrained(os.path.join(
                    output_dir, f"classification_head_{step}"))

            # evaluate model
            if step % args.eval_step == 0:
                logger.info(f"Evaluating model at step {step}")
                model.eval()
                with torch.no_grad():
                    true_pos = 0
                    true_neg = 0
                    false_pos = 0
                    false_neg = 0
                    total = 0
                    running_val_loss = 0
                    for val_data in val_loader:
                        val_embeddings = []
                        for input in val_data['inputs']:
                            output = model(input['input_ids'], input['attention_mask'])
                            val_embeddings.append(output['last_hidden_state'][:, 0, :])
                        val_embeddings = torch.cat(val_embeddings, dim=1)
                        val_logits = classification_head(val_embeddings)
                        val_loss = F.cross_entropy(val_logits, val_data['labels'])
                        running_val_loss += val_loss.item()
                        
                        # measure true positives, true negatives, false positives, false negatives
                        # use softmax to get probabilities
                        labels = val_data['labels']
                        probs = torch.softmax(val_logits, dim=1)
                        preds = torch.argmax(probs, dim=1)

                        true_pos += torch.sum((preds == 1) & (labels == 1)).item()
                        true_neg += torch.sum((preds == 0) & (labels == 0)).item()
                        false_pos += torch.sum((preds == 1) & (labels == 0)).item()
                        false_neg += torch.sum((preds == 0) & (labels == 1)).item()
                        total += len(labels)

                    # calculate accuracy, precision, recall, f1 score
                    accuracy = (true_pos + true_neg) / total
                    precision = true_pos / (true_pos + false_pos)
                    recall = true_pos / (true_pos + false_neg)
                    f1 = 2 * precision * recall / (precision + recall)
                    running_val_loss /= len(val_loader)
                    logger.info(f"Accuracy: {accuracy}")
                    logger.info(f"Precision: {precision}")
                    logger.info(f"Recall: {recall}")
                    logger.info(f"F1: {f1}")
                    logger.info(f"Validation loss: {running_val_loss}")
                    writer.add_scalar('val/accuracy', accuracy, step)
                    writer.add_scalar('val/precision', precision, step)
                    writer.add_scalar('val/recall', recall, step)
                    writer.add_scalar('val/f1', f1, step)
                    writer.add_scalar('val/loss', running_val_loss, step)
                model.train()

        scheduler.step()

    # close logger
    logger.info("Training finished")
    logger.removeHandler(fh)
    logger.removeHandler(console)
    fh.close()
    console.close()
