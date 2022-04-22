import sys
sys.path.append('../')

import os
import re
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import numpy as np


input_dir  = os.path.expanduser('~/.dataset/road/')
output_dir = '../output/road/'  # The output directory of parsing CAN data into sequences

with open(input_dir+'ambient/capture_metadata.json', 'r') as f:
    metadata_ambient = json.load(f)
with open(input_dir+'attacks/capture_metadata.json', 'r') as f:
    metadata_attacks = json.load(f)
    
log_files   = [[i for i in metadata_ambient],[i for i in metadata_attacks]]  # The input log files names


def log_to_csv(dataset, names, type="ambient",injection_data_str=None,injection_id=None,injection_interval=None):
    
    print(f"{input_dir}/{type}/{dataset}")
    input_path=f"{input_dir}/{type}/{dataset}.log"
    output_path=f"{output_dir}/structured/{type}/{dataset}.csv"
 
    df=pd.read_csv(input_path,sep="[ ,#]",names=names,engine="python")
    df.Time = df.Time.str.strip('()')
    df.drop(columns="Channel",inplace=True)
    df.Time=(df.Time).astype(float)
    df.Time=df.Time-df.Time[0]
    df['Label']=0
    
    if(type=="attacks"):
        if(dataset.split("_")[0]=="fuzzing"):
            df.loc[((df.Data==injection_data_str) & (df.Time.round(6) <= injection_interval[1]) & (df.Time.round(6) >=                              injection_interval[0])), 'Label'] = 1  
        else:
            modified_value=""
            for j in (injection_data_str):
                if(j!="X"):
                    modified_value+=j
            df.loc[((df.Data.apply(find, args=(injection_data_str,modified_value))) & (df.ID==injection_id) &                                       (df.Time.round(6) <= injection_interval[1]) & (df.Time.round(6) >= injection_interval[0])), 'Label'] = 1  

    df.to_csv(output_path,index=False)
    print(f"{dataset}.csv saved")

        
def stride(l, window_size=4, stride=2):
    d=[]
    seq_len=len(l)
    assert window_size<seq_len, "Get a window size smaller than sequence length"
    assert stride<seq_len, "Get a stride smaller than sequence length"
    
    for i in range(0,seq_len,stride):
        if (i+window_size>seq_len):
            break
        else:
            d.append(l[i:i+window_size])
    return d

def df_to_file(sequence, file_name):
    with open(file_name, 'w') as f:
        for i in sequence:
            f.write(' '.join([str(j) for j in i]))
            f.write('\n')

def find(x,f,modified):
    if(x.find(modified)==f.index(modified)):
        return True
    else:
        return False
            
            
def generate_id_sequence(dataset,type="ambient",seq_len=64,s=64,n=None,ratio=1):
    
    input_path=f"{output_dir}/structured/{type}/{dataset}.csv"
    output_path=f"{output_dir}/train_valid_test/"
    
    df=pd.read_csv(input_path)
    label=df["Label"].to_numpy()
    label=label.reshape((label.shape[0],1))
    print(np.unique(label, return_counts=True))
    
    
    #ID
    ID=df["ID"].to_numpy()
    list_ids=[]
    for i in ID:
        f=(int(i,16))
        list_ids.append(str(f))
        
    #Create sequences
    sequences=stride(list_ids, window_size=seq_len, stride=s)
    sequences=np.array(sequences)
    #Label sequences 
    labels=stride(list(label), window_size=seq_len, stride=s)
    sequences_labels= [int(np.any(i)) for i in labels]
    NumberOfInjections=[sum(i) for i in labels]
    sequences_labels=np.expand_dims(sequences_labels, axis=1)
    sequences_labels=np.array(sequences_labels)
    NumberOfInjections=np.expand_dims(NumberOfInjections, axis=1)
    NumberOfInjections=np.array(NumberOfInjections)
    print("Sequence Labels")
    print(np.unique(sequences_labels, return_counts=True))
    print("Number Of Injections")
    print(np.unique(NumberOfInjections, return_counts=True))
    
    if type=="attacks":
        attacks_abnormal=sequences[np.where(sequences_labels==1)[0]]
        print(len(attacks_abnormal))
        attacks_normal=sequences[np.where(sequences_labels==0)[0]]
        print(len(attacks_normal))
        #Save sequences
        df_to_file(attacks_abnormal, f"{output_path}/test/attacks/{dataset}")
        df_to_file(attacks_normal, f"{output_path}/test/ambient/{dataset}")
               
    else:
        ambient_normal = sequences[np.where(sequences_labels==0)[0]]
        np.random.shuffle(ambient_normal) #shuffle in place 
        ambient_normal_len= len(ambient_normal)
        train_len = n if n else int(ambient_normal_len * ratio)
        train = ambient_normal[:train_len]
        test_ambient_normal = ambient_normal[train_len:]
        #Save sequences
        df_to_file(train, f"{output_path}/train/{dataset}")
        df_to_file(test_ambient_normal,f"{output_path}/test/ambient/{dataset}")
        
    print("generate id sequences done")
    
if __name__ == "__main__":
    
    names=['Time','Channel','ID','Data']
    window_size=16 # CAN Sequence Length
    s=16 #Stride 
    print(names)
    for num,i in enumerate(log_files):
        if num==0:
            #Ambient 
            for k,j in enumerate(i):
                log_to_csv(j, names, type="ambient",injection_data_str=None,injection_id=None,injection_interval=None)
                generate_id_sequence(j,type="ambient",seq_len=window_size,s=s,n=None,ratio=1)
        else:
            #Attacks 
            for k,j in enumerate(i):
                if(k>=4): #consider only message injection attack 
                    injection_data_str=metadata_attacks[j]['injection_data_str']
                    iid=metadata_attacks[j]['injection_id'][2:].upper()
                    injection_id="0"+metadata_attacks[j]['injection_id'][2:].upper() if iid[0].isalpha() else iid
                    injection_interval=metadata_attacks[j]['injection_interval']
                    log_to_csv(j, names,                       type="attacks",injection_data_str=injection_data_str,injection_id=injection_id,injection_interval=injection_interval)
                    generate_id_sequence(j,type="attacks",seq_len=window_size,s=s,n=None,ratio=1)
        
    
