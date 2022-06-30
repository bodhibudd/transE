import random
import numpy as np
import os, sys, logging
import torch
import torch.nn as nn
from transE import TransE
from tqdm import tqdm
import torch.optim as optim
import time
class Trainer:
    def __init__(self, args, examples, dataloader, ent_num, rel_num):
        self.device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")
        self.args = args
        self.n_gpu = torch.cuda.device_count()
        if self.n_gpu > 0:
            torch.cuda.manual_seed(args.seed)
        self.model = TransE(args, ent_num, rel_num)
        self.model.to(self.device)
        if self.n_gpu > 1:
            self.model = nn.DataParallel(self.model.cuda(), device_ids=[0,1])
        self.train_dataloader, self.dev_dataloader = dataloader
        self.train_eval, self.dev_eval = examples
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

    def train(self):
        step_gap = 2000
        self.model.train()
        for epoch in range(self.args.epoch):
            global_loss = 0.0
            for step, batch in tqdm(enumerate(self.train_dataloader), mininterval=5, desc= "training at epoch : %d "%epoch, file=sys.stdout):
                batch_corrects, batch_corrupts = (t.to(self.device) for t in batch)
                loss = self.model(batch_corrects, batch_corrupts, is_eval=False)
                if self.n_gpu > 1:
                    loss = loss.mean()
                loss.backward()
                loss = loss.item()
                self.optimizer.step()
                self.optimizer.zero_grad()
                global_loss += loss
                if step % step_gap == 0:
                    current_loss = global_loss / step_gap
                    print(
                        "step {}/{} epoch {}, train loss {}".format(step, len(self.train_dataloader), epoch,
                                                                    current_loss))
                    global_loss = 0.0

            if epoch % 5 == 0:
                save_model = self.model.module if hasattr(self.model, "module") else self.model
                model_ouput_path = self.args.output+"pytorch_model.bin"
                torch.save(save_model.state_dict(), model_ouput_path)

    def test(self, en2ids, rel2ids, flit=True):
        self.model.eval()
        #加载模型
        model_path = self.args.output+"pytorch_model.bin"
        save_model = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(save_model)

        ent_hits = 0
        ent_rank_sum = 0
        rel_hits = 0
        rel_rank_sum = 0
        for j, batch in enumerate(self.dev_dataloader):
            batch = (t.to(self.device) for t in batch)
            for c_tuple in batch:
                rank_head = {}
                rank_tail = {}
                rank_rel = {}

                for ent in en2ids.keys():
                    cu_head_tuple = (en2ids[ent], c_tuple[1], c_tuple[2])
                    batch_correct_ids = torch.tensor([list(cu_head_tuple)], dtype=torch.long,device=self.device)
                    with torch.no_grad():
                        rank_head[cu_head_tuple] = self.model(batch_correct_ids, is_eval=True).mean()

                    cu_tail_tuple = (c_tuple[0], en2ids[ent], c_tuple[2])
                    batch_correct_ids = torch.tensor([cu_tail_tuple], dtype=torch.long,device=self.device)
                    with torch.no_grad():
                        rank_tail[cu_tail_tuple] = self.model(batch_correct_ids, is_eval=True).mean()
                rank_head = sorted(rank_head.items(), key= lambda x: x[1])
                rank_tail = sorted(rank_tail.items(), key = lambda x: x[1])
                for i in range(len(rank_head)):
                    if c_tuple[0] == rank_head[i][0][0]:
                        if i < 10:
                            ent_hits += 1
                            ent_rank_sum += (i+1)
                        break
                for i in range(len(rank_tail)):
                    if c_tuple[1] == rank_head[i][0][1]:
                        if i < 10:
                            ent_hits += 1
                            ent_rank_sum += (i+1)
                        break

                for rel in rel2ids.keys():
                    cu_rel_tuple = (c_tuple[0], c_tuple[1], rel2ids[rel])
                    batch_correct_ids = torch.tensor([list(cu_rel_tuple)], dtype=torch.long,device=self.device)
                    with torch.no_grad():
                        rank_rel[cu_rel_tuple] = self.model(batch_correct_ids, is_eval=True).mean()
                rank_rel = sorted(rank_rel.items(), key= lambda x: x[1])

                for i in range(len(rank_rel)):
                    if c_tuple[2] == rank_rel[i][0][2]:
                        if i < 10:
                            rel_hits += 1
                        rel_rank_sum +=(i+1)
                        break

            print("j: {}".format(j))
        self.hits10 = ent_hits / (2*len(self.dev_dataloader))
        self.mean_rank = ent_rank_sum / (2*len(self.dev_dataloader))
        self.relhits10 = rel_hits / len(self.dev_dataloader)
        self.relmean_rank = rel_rank_sum / len(self.dev_dataloader)
        print("hits@10 {},meanrank {}, relhits@10 {}, relmeanrank {}".format(self.hits10, self.mean_rank, self.relhits10, self.relmean_rank))






