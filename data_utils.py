import os, logging
import random
import pickle, sys
from torch.utils.data import Dataset, DataLoader
import torch
class Example:
    def __init__(self, correct_t, corrupt_t=None):
        self.correct_t = correct_t
        self.corrupt_t = corrupt_t

def read_example(filename, data_type="train"):
    examples = []
    with open(filename, 'r', encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            tri_t = tuple(line.split("	"))
            examples.append(Example(tri_t))
        logging.info("generating {} examples".format(data_type))
    return examples

def save(path, data):
    max_bytes = 2**31-1
    byte_out = pickle.dumps(data)
    byte_size = sys.getsizeof(byte_out)
    with open(path, 'wb') as f_out:
        for i in range(0, byte_size, max_bytes):
            f_out.write(byte_out[i:i+max_bytes])

def load(path):
    max_bytes = 2 ** 31 - 1
    byte_size = os.path.getsize(path)
    bytes_in = bytearray(0)
    with open(path, 'rb') as f_in:
        for i in range(0, byte_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
        obj = pickle.loads(bytes_in)
    return obj

class TripleDataset(Dataset):
    def __init__(self, datas, ent2ids, rel2ids, data_type="train"):
        self.datas = datas
        self.data_type = data_type
        self.ent2ids = ent2ids
        self.rel2ids = rel2ids
        self.is_train = True if data_type == 'train' else False

    def __len__(self):
        return len(self.datas)
    def __getitem__(self, idx):
        return self.datas[idx]

    def collate_fn(self, examples):
        batch = []
        for example in examples:
            correct_t = example.correct_t
            if self.is_train:
                #对于每一个正样本，生成一个负样本
                p = random.random()
                #大于0.5，随机替换头, 否则，随机替换尾
                if p > 0.5:
                    head = correct_t[0]
                    while(head == correct_t[0]):
                        head = random.sample(self.ent2ids.keys(),1)[0]
                    corrupt_t = (head, correct_t[1], correct_t[2])
                else:
                    tail = correct_t[1]
                    while(tail == correct_t[1]):
                        tail = random.sample(self.ent2ids.keys(),1)[0]
                    corrupt_t = (correct_t[0], tail, correct_t[2])
                batch.append((correct_t, corrupt_t))
            else:
                batch.append(correct_t)
        #映射id
        batch_correct_ids = []
        batch_corrupt_ids = []
        if self.is_train:
            for c_t, cu_t in batch:
                c_head_id, c_tail_id, c_rel_id = self.ent2ids[c_t[0]], self.ent2ids[c_t[1]], self.rel2ids[c_t[2]]
                cu_head_id, cu_tail_id, cu_rel_id = self.ent2ids[cu_t[0]], self.ent2ids[cu_t[1]], self.rel2ids[cu_t[2]]
                batch_correct_ids.append((c_head_id, c_tail_id, c_rel_id))
                batch_corrupt_ids.append((cu_head_id, cu_tail_id, cu_rel_id))
            batch_correct_ids = torch.tensor(batch_correct_ids, dtype=torch.long)
            batch_corrupt_ids = torch.tensor(batch_corrupt_ids, dtype=torch.long)
            return batch_correct_ids, batch_corrupt_ids
        else:
            for c_t in batch:
                c_head_id, c_tail_id, c_rel_id = self.ent2ids[c_t[0]], self.ent2ids[c_t[1]], self.rel2ids[c_t[2]]
                batch_correct_ids.append((c_head_id, c_tail_id, c_rel_id))
            batch_correct_ids = torch.tensor(batch_correct_ids, dtype=torch.long)
            return batch_correct_ids

    def get_dataloader(self, batch_size):
        return DataLoader(self, batch_size=batch_size, collate_fn=self.collate_fn)





