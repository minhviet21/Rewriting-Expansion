from torch import nn
import torch
import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModel
from data.dataset import TopiocqaDataset
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW
from tqdm import tqdm, trange
from os.path import join as oj
import shutil
import os

def check_dir_exist_or_build(dir_list, erase_dir_content = None):
    for x in dir_list:
        if not os.path.exists(x):
            os.makedirs(x)
    if erase_dir_content:
        for dir_path in erase_dir_content:
            shutil.rmtree(dir_path)
            os.makedirs(dir_path)

def save_model(args, model, query_tokenizer, save_model_order, epoch, step, loss):
    output_dir = oj(args.model_output_path, '{}-{}-best-model'.format("KD-ANCE-prefix", args.decode_type))
    check_dir_exist_or_build([output_dir])
    model_to_save = model.module if hasattr(model, 'module') else model
    #model_to_save.t5.save_pretrained(output_dir)
    model_to_save.save_pretrained(output_dir)
    query_tokenizer.save_pretrained(output_dir)
    print("Step {}, Save checkpoint at {}".format(step, output_dir))

def loss(query_embed, passage_embed):
    loss_function = nn.MSELoss()
    return loss_function(query_embed, passage_embed)
 
def get_optimizer(args, model: nn.Module, weight_decay: float = 0.0, ) -> torch.optim.Optimizer:
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    return optimizer

def train():
    passage_tokenizer = AutoTokenizer.from_pretrained(args.passage_encoder)
    passage_encoder = AutoModel.from_pretrained(args.passage_encoder).to(args.device)
   
    query_tokenizer = T5Tokenizer.from_pretrained(args.query_encoder)
    query_encoder = T5ForConditionalGeneration.from_pretrained(args.query_encoder).to(args.device)

    total_training_steps = args.num_train_epochs * (len(train_dataset) // args.batch_size + int(bool(len(train_dataset) % args.batch_size)))
    num_warmup_steps = args.num_warmup_portion * total_training_steps

    train_dataset = TopiocqaDataset(args, query_tokenizer, args.train_file_path)
    train_loader = DataLoader(train_dataset,
                              batch_size = args.batch_size,
                              shuffle=True, 
                              collate_fn=train_dataset.get_collate_fn(args))
    
    optimizer = get_optimizer(args, query_encoder, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_training_steps)

    global_step = 0
    save_model_order = 0
    
    num_steps_per_epoch = total_training_steps // args.num_train_epochs

    if isinstance(args.print_steps, float):
        args.print_steps = int(args.print_steps * num_steps_per_epoch)
        args.print_steps = max(1, args.print_steps)

    epoch_iterator = trange(args.num_train_epochs, desc="Epoch", disable=args.disable_tqdm)

    best_loss = 1000
    for epoch in epoch_iterator:
        query_encoder.train()
        passage_encoder.eval()
        for batch in tqdm(train_loader,  desc="Step", disable=args.disable_tqdm):
            query_encoder.zero_grad()

            bt_conv_query = batch['bt_input_ids'].to(args.device)
            bt_conv_query_mask = batch['bt_attention_mask'].to(args.device)
            bt_pos_docs = batch['bt_pos_docs'].to(args.device) 
            bt_pos_docs_mask = batch['bt_pos_docs_mask'].to(args.device)
            bt_neg_docs = batch['bt_neg_docs'].to(args.device)
            bt_neg_docs_mask = batch['bt_neg_docs_mask'].to(args.device)
            bt_oracle_query = batch['bt_labels'].to(args.device)
            
            output = query_encoder(input_ids=bt_conv_query, 
                         attention_mask=bt_conv_query_mask, 
                         labels=bt_oracle_query)
            decode_loss = output.loss  # B * dim
            conv_query_embs = output.encoder_last_hidden_state[:, 0]

            with torch.no_grad():
                pos_doc_embs = passage_encoder(bt_pos_docs, bt_pos_docs_mask).detach()

            ranking_loss = loss(conv_query_embs, pos_doc_embs)
            loss = decode_loss + args.alpha * ranking_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(query_encoder.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            if args.print_steps > 0 and global_step % args.print_steps == 0:
                print("Epoch = {}, Global Step = {}, ranking loss = {}, decode loss = {}, total loss = {}".format(
                                epoch + 1,
                                global_step,
                                ranking_loss.item(),
                                decode_loss.item(),
                                loss.item()))
                
            global_step += 1   

            if best_loss > loss:
                save_model(args, query_encoder, query_tokenizer, save_model_order, epoch, global_step, loss.item())
                best_loss = loss
                print("Epoch = {}, Global Step = {}, ranking loss = {}, decode loss = {}, total loss = {}".format(
                                epoch + 1,
                                global_step,
                                ranking_loss.item(),
                                decode_loss.item(),
                                loss.item()))
                
    print("Training finish!") 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_encoder", type=str, default="")
    parser.add_argument("--passage_encoder", type=str, default="")

    parser.add_argument("--train_file_path", type=str, default="")
    parser.add_argument("--log_dir_path", type=str, default="")
    parser.add_argument('--model_output_path', type=str, default="")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--collate_fn_type", type=str, default="flat_concat_for_train")
    parser.add_argument("--decode_type", type=str, default="oracle")
    parser.add_argument("--use_prefix", type=bool, default=True)

    parser.add_argument("--per_gpu_train_batch_size", type=int,  default=8)
    parser.add_argument("--num_train_epochs", type=int, default=15, help="num_train_epochs")
    parser.add_argument("--max_query_length", type=int, default=32, help="Max single query length")
    parser.add_argument("--max_doc_length", type=int, default=384, help="Max doc length")
    parser.add_argument("--max_response_length", type=int, default=64, help="Max response length")
    parser.add_argument("--max_concat_length", type=int, default=512, help="Max concatenation length of the session")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--disable_tqdm", type=bool, default=True)

    parser.add_argument("--print_steps", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--num_warmup_portion", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    return args

if __name__ == '__main__':
    args = get_args()