from torch.utils.data import Dataset
import random
from tqdm import tqdm, trange
import json
import torch

class TopiocqaDataset(Dataset):
    def __init__(self, args, tokenizer, filename):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()

        i = 0
        for line in tqdm(data):
            record = json.loads(line)
            flat_concat = []

            ctx_utts_text = []
            cur_utt_text = record['query']
            history_query = record['history_query']
            history_answer = record['history_answer']
            for i in range(len(history_query)):
                ctx_utts_text.append(history_query[i])
                ctx_utts_text.append(history_answer[i])
            cur_response_text = record["answer"]
            oracle_utt_text = record["rewrite"]
            
            if args.use_prefix:
                cur_utt_text = "Câu hỏi: " + cur_utt_text
                first_context = True
                
            cur_utt = tokenizer.encode(cur_utt_text, add_special_tokens = True, max_length = args.max_query_length)
            flat_concat.extend(cur_utt)
            for j in range(len(ctx_utts_text) - 1, -1, -1):
                if j % 2 == 1:
                    max_length = args.max_response_length
                else:
                    max_length = args.max_query_length
                if args.use_prefix and first_context:
                    ctx_utts_text[j] = "Ngữ cảnh: " + ctx_utts_text[j]
                    first_context = False
                utt = tokenizer.encode(ctx_utts_text[j], add_special_tokens=True, max_length=max_length, truncation=True) # not remove [CLS]
                if len(flat_concat) + len(utt) > args.max_concat_length:
                    flat_concat += utt[:args.max_concat_length - len(flat_concat) - 1] + [utt[-1]]    # must ended with [SEP]
                    break
                else:
                    flat_concat.extend(utt) 
                    
            flat_concat, flat_concat_mask = padding_seq_to_same_length(flat_concat, max_pad_length = args.max_concat_length)

            if args.collate_fn_type == "flat_concat_for_train":
                if args.decode_type == "oracle":
                    target_seq = oracle_utt_text
                    target_encoding = tokenizer(target_seq, padding="max_length", max_length=args.max_query_length, truncation=True)    
                elif args.decode_type == "answer":
                    target_seq = cur_response_text
                    target_encoding = tokenizer(target_seq, padding="max_length", max_length=args.max_response_length, truncation=True)    

                labels = target_encoding.input_ids
                labels = torch.tensor(labels)
                labels[labels == tokenizer.pad_token_id] = -100
                labels = labels.tolist()
                
                pos_docs = []
                pos_docs.extend(tokenizer.encode(record["pos_docs"], add_special_tokens=True, max_length = args.max_doc_length))
                pos_docs, pos_docs_mask = padding_seq_to_same_length(pos_docs, max_pad_length = args.max_doc_length)

                self.examples.append([record['id'], 
                                flat_concat,
                                flat_concat_mask,
                                labels,
                                cur_utt_text,
                                oracle_utt_text,
                                pos_docs,
                                pos_docs_mask])
                i += 1
            else:
                labels = []
                pos_docs = []
                self.examples.append([record['id'], 
                                        flat_concat,
                                        flat_concat_mask,
                                        labels,
                                        cur_utt_text,
                                        oracle_utt_text,
                                        pos_docs,
                                        pos_docs_mask])

    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_input_ids": [],
                             "bt_attention_mask": [],
                             "bt_labels": [],
                             "bt_cur_utt_text": [],
                             "bt_oracle_utt_text": [],
                             "bt_pos_docs": [],
                             "bt_pos_docs_mask": [],
                            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_input_ids"].append(example[1])
                collated_dict["bt_attention_mask"].append(example[2])
                collated_dict["bt_labels"].append(example[3])
                collated_dict["bt_cur_utt_text"].append(example[4])
                collated_dict["bt_oracle_utt_text"].append(example[5])
                collated_dict["bt_pos_docs"].append(example[6])
                collated_dict["bt_pos_docs_mask"].append(example[7])

            not_need_to_tensor_keys = set(["bt_sample_ids", "bt_cur_utt_text", "bt_oracle_utt_text"])
            if args.collate_fn_type == "flat_concat_for_test":
                not_need_to_tensor_keys.add("bt_labels")

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
                    
            return collated_dict

        return collate_fn
    
def padding_seq_to_same_length(input_ids, max_pad_length, pad_token = 0):
    padding_length = max_pad_length - len(input_ids)
    padding_ids = [pad_token] * padding_length
    attention_mask = []

    if padding_length <= 0:
        attention_mask = [1] * max_pad_length
        input_ids = input_ids[:max_pad_length]
    else:
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        input_ids = input_ids + padding_ids
            
    assert len(input_ids) == max_pad_length
    assert len(attention_mask) == max_pad_length
  
    return input_ids, attention_mask