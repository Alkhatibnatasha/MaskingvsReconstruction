import sys
sys.path.append('../')

import os
import re
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import glob 


input_dir  = os.path.expanduser('~/.dataset/IVNID/car_track_preliminary_train')
output_dir = '../output/ivnid/'  # The output directory of parsing CAN data into sequences

all_files=glob.glob(f"{input_dir}/*")
train_files = glob.glob(f"{input_dir}/Attack_free_*")
test_files=[]
for i in all_files:
    if i not in train_files:
        test_files.append(i)
        
        
def preprocess(input_path, dataset,car="CHEVROLET_Spark",type="ambient"):
    
    names=['Time','ID','DLC','Data','Label']
    print(f"{input_path}")
    output_path=f"{output_dir}/{car}/structured/{type}/{dataset}"
    
    if(type=="ambient"):
        df=pd.read_csv(input_path,names=names)
        df["Label"]=0
        print(df["Label"].value_counts())
    else :
        df=pd.read_csv(input_path,names=names)
        df["Label"]=df["Label"].apply(lambda x:0 if x=="R" else 1)
        print(df["Label"].value_counts())
        
    df.to_csv(output_path,index=False)
    print(f"{dataset} saved")
    

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
        
def generate_id_sequence(dataset,car,type="ambient",seq_len=64,s=64,n=None,ratio=1):
    
    input_path=f"{output_dir}/{car}/structured/{type}/{dataset}"
    output_path=f"{output_dir}/{car}/train_test/"
    
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
        dataset_name=dataset.split(".")[0]
        df_to_file(attacks_abnormal, f"{output_path}/test/attacks/{dataset_name}")
        df_to_file(attacks_normal, f"{output_path}/test/ambient/{dataset_name}")
               
    else:
        ambient_normal = sequences[np.where(sequences_labels==0)[0]]
        np.random.shuffle(ambient_normal) #shuffle in place 
        ambient_normal_len= len(ambient_normal)
        train_len = n if n else int(ambient_normal_len * ratio)
        train = ambient_normal[:train_len]
        test_ambient_normal = ambient_normal[train_len:]
        #Save sequences
        dataset_name=dataset.split(".")[0]
        df_to_file(train, f"{output_path}/train/{dataset_name}")
        df_to_file(test_ambient_normal,f"{output_path}/test/ambient/{dataset_name}")
        
    print("generate id sequences done")
        
if __name__ == "__main__":
    
    cars=["CHEVROLET_Spark", "HYUNDAI_Sonata", "KIA_Soul"] # attacks on 3 types of vehicles
    window_size=16 # CAN Sequence Length
    s=16 #Stride 
    
    for i in train_files: 
        dataset=i.split("/")[-1]
        car="_".join(dataset.split("_")[2:4])
        preprocess(i, dataset,car,type="ambient") #Normal training of data 
        generate_id_sequence(dataset,car,type="ambient",seq_len=window_size,s=s,n=None,ratio=1)
        
    for i in test_files: 
        dataset=i.split("/")[-1]
        car="_".join(dataset.split("_")[1:3])
        preprocess(i, dataset,car,type="attacks") #Normal training of data
        generate_id_sequence(dataset,car,type="attacks",seq_len=window_size,s=s,n=None,ratio=1)
        
    
   
    



