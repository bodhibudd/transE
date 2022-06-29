import torch
import numpy as np
import argparse,os, random
from data_utils import read_example, save, load, TripleDataset
from train import Trainer
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="../data/")
    parser.add_argument("--output", type=str, default="output/")
    parser.add_argument("--mode", type=str, default="dev")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--patient_stop", type=str, default=1000)
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--norm", type=int, default=2)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()

#构建样本集
def build_dataset(args, ent2ids, rel2ids):
    train_src_path = args.input+ "train.txt"
    dev_src_path = args.input + "valid.txt"
    if not os.path.exists(train_src_path) and not os.path.exists(dev_src_path):
        raise Exception("文件不存在")
    train_examples_path = args.input+"train_examples.pkl"
    dev_examples_path = args.input + "dev_examples.pkl"
    if not os.path.exists(train_examples_path):
        train_examples = read_example(train_src_path, data_type="train")
        dev_examples = read_example(dev_src_path, data_type="dev")
        #保存
        save(train_examples_path, train_examples)
        save(dev_examples_path, dev_examples)
    else:
        train_examples = load(train_examples_path)
        dev_examples = load(dev_examples_path)
    train_dataset, dev_dataset = TripleDataset(train_examples, ent2ids, rel2ids, data_type="train"), \
                                 TripleDataset(dev_examples, ent2ids, rel2ids, data_type="dev")
    dataloader = train_dataset.get_dataloader(args.batch_size), dev_dataset.get_dataloader(args.batch_size)
    eval_examples = train_examples, dev_examples
    return eval_examples, dataloader

if __name__ == "__main__":
    args = get_args()
    #固定随机种子
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    #加载实体关系id映射
    ent2ids = {}
    rel2ids = {}
    with open("../data/entity2id.txt", 'r', encoding="utf8") as f, open("../data/relation2id.txt", 'r', encoding="utf8") as g:
        for i, line in enumerate(f):

            l = line.split("	")
            ent2ids[l[0].strip()] = int(l[1].strip())

        for i, line in enumerate(g):

            l = line.split("	")
            rel2ids[l[0].strip()] = int(l[1].strip())

    eval_examples, dataloader = build_dataset(args, ent2ids, rel2ids)
    train = Trainer(args, eval_examples, dataloader, len(ent2ids), len(rel2ids))
    if args.mode == "train":
        train.train()
    else:
        train.test(ent2ids, rel2ids)