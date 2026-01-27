import os
import math
from data_set import MyDataset
from torch.utils.data import DataLoader
import torch
import logging
from tqdm import tqdm, trange
from sklearn import metrics
import wandb

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def _schedule_lambda(start, end, epoch_idx, warmup_epochs, ramp_epochs, schedule):
    if schedule == "none":
        return end
    if epoch_idx < warmup_epochs:
        return start
    if ramp_epochs <= 0:
        return end
    t = (epoch_idx - warmup_epochs + 1) / float(ramp_epochs)
    t = max(0.0, min(1.0, t))
    if schedule == "cosine":
        t = 0.5 * (1.0 - math.cos(math.pi * t))
    return start + (end - start) * t


def train(args, model,device, train_data, dev_data, test_data, processor):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    train_loader = DataLoader(dataset=train_data,
                              batch_size=args.train_batch_size,
                              collate_fn=MyDataset.collate_func,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    total_steps = int(len(train_loader) * args.num_train_epochs)
    model.to(device, non_blocking=True)

    # Freeze CLIP if specified (for stage 1 training)
    if hasattr(args, 'freeze_clip') and args.freeze_clip:
        if hasattr(model, 'model'):
            for param in model.model.parameters():
                param.requires_grad = False
            print("✅ CLIP model frozen for stage 1 training")
            # Count frozen parameters
            frozen_params = sum(p.numel() for p in model.model.parameters())
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  - Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
            print(f"  - Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")

    if args.optimizer_name == 'adafactor':
        from transformers.optimization import Adafactor, AdafactorSchedule

        print('Use Adafactor Optimizer for Training.')
        optimizer = Adafactor(
            model.parameters(),
            lr=None,
            weight_decay=args.weight_decay,
            relative_step=True,
            scale_parameter=True,
            warmup_init=True
        )
        scheduler = AdafactorSchedule(optimizer)
    elif args.optimizer_name == 'adam':
        print('Use AdamW Optimizer for Training.')
        from transformers.optimization import AdamW, get_linear_schedule_with_warmup
        if args.model == 'RCLMuFN':
            # BERT/ResNet branches are disabled.
            # for param in model.bert_model.parameters():
            #     param.requires_grad = False
            # for param in model.backbone.parameters():
            #     param.requires_grad = False

            clip_params = list(map(id, model.model.parameters()))
            # resnet_params = list(map(id, model.backbone.parameters()))
            # bert_params = list(map(id, model.bert_model.parameters()))
            # base_params = filter(lambda p: id(p) not in clip_params and id(p) not in bert_params
            #                      and id(p) not in resnet_params, model.parameters())
            base_params = filter(lambda p: id(p) not in clip_params, model.parameters())

            # Create optimizer: if CLIP is frozen, only optimize base params
            if hasattr(args, 'freeze_clip') and args.freeze_clip:
                optimizer = AdamW([{"params": base_params}],
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
            else:
                optimizer = AdamW([
                        {"params": base_params},
                        {"params": model.model.parameters(),"lr": args.clip_learning_rate},
                        ], lr=args.learning_rate, weight_decay=args.weight_decay)

            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                    num_training_steps=total_steps)
        else:
            optimizer = optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=args.weight_decay)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                num_training_steps=total_steps)
    else:
        raise Exception('Wrong Optimizer Name!!!')

    max_test_acc = 0.

    diag_interval = 50
    for i_epoch in trange(0, int(args.num_train_epochs), desc="Epoch", disable=False):
        # Set current epoch for temperature scheduling in CID module
        if hasattr(model, 'set_epoch'):
            model.set_epoch(i_epoch)

        # Schedule lambda weights for CID losses
        model.lambda_ratio = _schedule_lambda(
            args.lambda_ratio_start,
            args.lambda_ratio_end,
            i_epoch,
            args.lambda_warmup_epochs,
            args.lambda_ramp_epochs,
            args.lambda_schedule
        )
        model.lambda_itm = _schedule_lambda(
            args.lambda_itm_start,
            args.lambda_itm_end,
            i_epoch,
            args.lambda_warmup_epochs,
            args.lambda_ramp_epochs,
            args.lambda_schedule
        )

        # Log scheduled parameters
        log_dict = {"lambda_ratio": model.lambda_ratio, "lambda_itm": model.lambda_itm}
        if hasattr(model, 'cid') and hasattr(model.cid, 'current_tau'):
            log_dict["tau"] = model.cid.current_tau.item()
        wandb.log(log_dict)
        sum_loss = 0.
        sum_step = 0

        iter_bar = tqdm(train_loader, desc="Iter (loss=X.XXX)", disable=False)
        model.train()

        for step, batch in enumerate(iter_bar):
            text_list, image_list, label_list, id_list = batch
            if args.model == 'RCLMuFN':
                inputs = processor(text=text_list, images=image_list, padding='max_length',
                                   truncation=True, max_length=args.max_len, return_tensors="pt").to(device)
                labels = torch.tensor(label_list).to(device, non_blocking=True)

            '''add==============='''
            loss, score = model(inputs, batch, labels=labels,)
            sum_loss += loss.item()
            sum_step += 1

            iter_bar.set_description("Iter (loss=%5.3f)" % loss.item())
            if hasattr(model, "last_cid_stats") and step % diag_interval == 0:
                wandb.log(model.last_cid_stats)
            loss.backward()
            optimizer.step()
            if args.optimizer_name == 'adam':
                scheduler.step() 
            optimizer.zero_grad()

        wandb.log({'train_loss': sum_loss/sum_step})
        dev_acc, dev_f1 ,dev_precision,dev_recall = evaluate_acc_f1(args, model, device, dev_data, processor, mode='dev')
        wandb.log({'dev_acc': dev_acc, 'dev_f1': dev_f1, 'dev_precision': dev_precision, 'dev_recall': dev_recall})
        logging.info("i_epoch is {}, dev_acc is {}, dev_f1 is {}, dev_precision is {}, dev_recall is {}".format(i_epoch, dev_acc, dev_f1, dev_precision, dev_recall))

        test_acc, test_f1, test_precision, test_recall = evaluate_acc_f1(args, model, device, test_data, processor, macro=True, mode='test')
        _, test_f1_, test_precision_, test_recall_ = evaluate_acc_f1(args, model, device, test_data, processor, mode='test')
        wandb.log({'test_acc': test_acc, 'macro_test_f1': test_f1,
                 'macro_test_precision': test_precision,'macro_test_recall': test_recall, 'micro_test_f1': test_f1_,
                 'micro_test_precision': test_precision_,'micro_test_recall': test_recall_})
        logging.info("i_epoch is {}, test_acc is {}, macro_test_f1 is {}, macro_test_precision is {}, macro_test_recall is {}, micro_test_f1 is {}, micro_test_precision is {}, micro_test_recall is {}".format(i_epoch, test_acc, test_f1, test_precision, test_recall, test_f1_, test_precision_, test_recall_))

        if test_acc > max_test_acc:
            max_test_acc = test_acc
            path_to_save = os.path.join(args.output_dir, args.model)
            if not os.path.exists(path_to_save):
                os.mkdir(path_to_save)
            model_to_save = (model.module if hasattr(model, "module") else model)
            torch.save(model_to_save.state_dict(), os.path.join(path_to_save, 'model.pt'))
            logging.info("New best test_acc {:.6f} at epoch {}, saved model.".format(max_test_acc, i_epoch))

        torch.cuda.empty_cache()
    logger.info('Train done')


def evaluate_acc_f1(args, model, device, data, processor, macro=False,pre = None, mode='test'):
        data_loader = DataLoader(data, batch_size=args.dev_batch_size, collate_fn=MyDataset.collate_func, shuffle=False,
                                 num_workers=4, pin_memory=True)
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None

        model.eval()
        sum_loss = 0.
        sum_step = 0
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                text_list, image_list, label_list, id_list = t_batch
                if args.model == 'RCLMuFN':
                    inputs = processor(text=text_list, images=image_list, padding='max_length',
                                       truncation=True, max_length=args.max_len, return_tensors="pt").to(device)
                    labels = torch.tensor(label_list).to(device, non_blocking=True)

                t_targets = labels
                loss, t_outputs = model(inputs, t_batch, labels=labels)
                sum_loss += loss.item()
                sum_step += 1
  
                outputs = torch.argmax(t_outputs, -1)

                n_correct += (outputs == t_targets).sum().item()
                n_total += len(outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, outputs), dim=0)
        if mode == 'test':
            wandb.log({'test_loss': sum_loss/sum_step})
        else:
            wandb.log({'dev_loss': sum_loss/sum_step})
        if pre != None:
            with open(pre,'w',encoding='utf-8') as fout:
                predict = t_outputs_all.cpu().numpy().tolist()
                label = t_targets_all.cpu().numpy().tolist()
                for x,y,z in zip(predict,label):
                    fout.write(str(x) + str(y) +z+ '\n')
        if not macro:   
            acc = n_correct / n_total
            f1 = metrics.f1_score(t_targets_all.cpu(), t_outputs_all.cpu())
            precision =  metrics.precision_score(t_targets_all.cpu(),t_outputs_all.cpu())
            recall = metrics.recall_score(t_targets_all.cpu(),t_outputs_all.cpu())
        else:
            acc = n_correct / n_total
            f1 = metrics.f1_score(t_targets_all.cpu(), t_outputs_all.cpu(), labels=[0, 1],average='macro')
            precision =  metrics.precision_score(t_targets_all.cpu(),t_outputs_all.cpu(), labels=[0, 1],average='macro')
            recall = metrics.recall_score(t_targets_all.cpu(),t_outputs_all.cpu(), labels=[0, 1],average='macro')
        return acc, f1 ,precision,recall
