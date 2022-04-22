import sys
sys.path.append("../")
sys.path.append("../../")

import os

import argparse
import torch

from bert_pytorch.dataset import WordVocab
from bert_pytorch import Predictor,Trainer
from bert_pytorch.dataset.utils import seed_everything


import glob 

options = dict()
options['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
options['car'] = "CHEVROLET_Spark/"
options["output_dir"] = "../output/ivnid/" + options['car']
options["model_dir"] = options["output_dir"] + "bert/"
options["model_path"] = options["model_dir"] + "best_bert.pth"

train_vocab_ambient = glob.glob(options["output_dir"]+"train_test/train/*")
test_vocab_attacks = glob.glob(options["output_dir"]+"train_test/test/attacks/*")
test_vocab_ambient = glob.glob(options["output_dir"]+"train_test/test/ambient/*")
vocab_files=[*train_vocab_ambient,*test_vocab_attacks,*test_vocab_ambient]
options["train_vocab"] = options["output_dir"] + "train_vocab"

with open(options["train_vocab"], 'w') as outfile:
    for names in vocab_files:
        with open(names) as infile:
            outfile.write(infile.read())
            

options["vocab_path"] = options["output_dir"] + "vocab.pkl"  # pickle file

train_files = glob.glob(options["output_dir"]+"train_valid_test/train/*")
options["train_data"] = options["output_dir"] + "train"
#Concatenate for train
with open(options["train_data"], 'w') as outfile:
    for names in train_files:
        with open(names) as infile:
            outfile.write(infile.read())
            
  
options["window_size"] = 16
options["adaptive_window"] = True
options["seq_len"] = 16
options["max_len"] = 16 # for position embedding
options["min_len"] = 16
options["mask_ratio"] = 0.15
# sample ratio
options["train_ratio"] = 1
options["valid_ratio"] = 0.1
options["test_ratio"] = 1

# features
options["is_id"] = True
options["is_time"] = False

options["hypersphere_loss"] = True
options["hypersphere_loss_test"] = False

options["scale"] = None # MinMaxScaler()
options["scale_path"] = options["model_dir"] + "scale.pkl"

# model
options["hidden"] = 256 # embedding size
options["layers"] = 4
options["attn_heads"] = 4

options["epochs"] = 200
options["n_epochs_stop"] = 10
options["batch_size"] = 32

options["corpus_lines"] = None
options["on_memory"] = True
options["num_workers"] = 0
options["lr"] = 1e-3
options["adam_beta1"] = 0.9
options["adam_beta2"] = 0.999
options["adam_weight_decay"] = 0.00
options["with_cuda"]= True
options["cuda_devices"] = None
options["can_freq"] = None

# predict
options["num_candidates"] = 6
options["gaussian_mean"] = 0
options["gaussian_std"] = 1

seed_everything(seed=1234)

if not os.path.exists(options['model_dir']):
    os.makedirs(options['model_dir'], exist_ok=True)

print("device", options["device"])
print("features id:{} time: {}\n".format(options["is_id"], options["is_time"]))
print("mask ratio", options["mask_ratio"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    
    
    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(mode='train')
    
    predict_parser = subparsers.add_parser('predict')
    predict_parser.set_defaults(mode='predict')
    predict_parser.add_argument("-m", "--mean", type=float, default=0)
    predict_parser.add_argument("-s", "--std", type=float, default=1)


    vocab_parser = subparsers.add_parser('vocab')
    vocab_parser.set_defaults(mode='vocab')
    vocab_parser.add_argument("-s", "--vocab_size", type=int, default=None)
    vocab_parser.add_argument("-e", "--encoding", type=str, default="utf-8")
    vocab_parser.add_argument("-m", "--min_freq", type=int, default=1)

    args = parser.parse_args()
    print("arguments", args)
    
    if args.mode == 'train':
        print(options)
        Trainer(options).train()
        
    elif args.mode == 'predict':
        Predictor(options).predict()

    elif args.mode == 'vocab':
        with open(options["train_vocab"], "r", encoding=args.encoding) as f:
            texts = f.readlines()
        vocab = WordVocab(texts, max_size=args.vocab_size, min_freq=args.min_freq)
        print(options["train_vocab"] )
        print("VOCAB SIZE:", len(vocab))
        print("save vocab in", options["vocab_path"])
        vocab.save_vocab(options["vocab_path"])