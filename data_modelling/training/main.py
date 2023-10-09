import argparse
import logging
import os
import time

import multiprocess
import torch
import torch.nn.functional as F
import torch.optim as optim
from accelerate import Accelerator
from accelerate.logging import get_logger
from data import WikiDataset
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel, AutoTokenizer

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
    parser.add_argument('--ga_steps', type=int, default=1,
                        help='Number of steps for gradient accumulation')
    parser.add_argument('--full_freeze_epochs', type=int, default=0,
                        help='Number of epochs to freeze all lazyers except classification head')
    parser.add_argument('--freeze_layers', type=int, default=0,
                        help='Number of initial layers to freeze')
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
        os.makedirs('runs', exist_ok=True)
    date_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    tb_dir = os.path.join('runs', date_time)
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir, exist_ok=True)
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

    logger.info("Loading datasets")
    train_set = WikiDataset(args.data_dir, 'train')
    val_set = WikiDataset(args.data_dir, 'val')
    logger.info(f"Train set size: {len(train_set)}")
    logger.info(f"Validation set size: {len(val_set)}")

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

    # store model weights to keep track of model distance
    model_weights = torch.cat([param.data.flatten()
                              for param in model.parameters()])

    logger.info("Initializing optimizer")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=args.gamma_lr)

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
        return {'sources': sources, 'contexts': contexts, 'targets': targets, 'labels': labels}

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

    # prepare all objects with accelerator
    model, classification_head, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, classification_head, optimizer, train_loader, val_loader, scheduler)

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

    logger.info("Starting training")
    step = 0
    running_loss = 0
    for epoch in range(args.num_epochs):
        for data in train_loader:
            with accelerator.accumulate(model):
                step += 1
                embeddings = []
                for input in ['sources', 'contexts', 'targets']:
                    output = model(data[input], data[input])
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
                    writer.add_scalar(
                        'train/loss', running_loss / args.print_step, step)
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
                torch.save(accelerator.unwrap_model(classification_head).state_dict(), os.path.join(
                    output_dir, f"classification_head_{step}.pth"))

            # evaluate model
            if step % args.eval_step == 0:
                logger.info(f"Evaluating model at step {step}")
                model.eval()
                with torch.no_grad():
                    # compare current model weights to initial model weights
                    current_model_weights = torch.cat(
                        [param.data.flatten() for param in model.parameters()])
                    model_distance = torch.norm(
                        current_model_weights - model_weights)

                    true_pos = 0
                    true_neg = 0
                    false_pos = 0
                    false_neg = 0
                    total = 0
                    running_val_loss = 0
                    for val_data in val_loader:
                        val_embeddings = []
                        for input in ['sources', 'contexts', 'targets']:
                            output = model(
                                val_data[input], val_data[input])
                            val_embeddings.append(
                                output['last_hidden_state'][:, 0, :])
                        val_embeddings = torch.cat(val_embeddings, dim=1)
                        val_logits = classification_head(val_embeddings)
                        val_loss = F.cross_entropy(
                            val_logits, val_data['labels'])
                        running_val_loss += val_loss.item()

                        # measure true positives, true negatives, false positives, false negatives
                        # use softmax to get probabilities
                        labels = val_data['labels']
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
                    writer.add_scalar('val/accuracy', accuracy, step)
                    writer.add_scalar('val/precision', precision, step)
                    writer.add_scalar('val/recall', recall, step)
                    writer.add_scalar('val/f1', f1, step)
                    writer.add_scalar('val/loss', running_val_loss, step)
                    writer.add_scalar('model/distance', model_distance, step)
                model.train()

        scheduler.step()

        # unfreeze model if necessary
        if epoch + 1 == args.full_freeze_epochs:
            logger.info(
                f"Unfreezing model except first {args.freeze_layers} layers")
            for param in model.parameters():
                param.requires_grad = True
            for param in model.base_model.embeddings.parameters():
                param.requires_grad = False
            for param in model.base_model.encoder.layer[:args.freeze_layers].parameters():
                param.requires_grad = False

    # close logger
    logger.info("Training finished")
    writer.close()
