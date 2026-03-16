import os,random
import numpy as np
import sentencepiece as spm

TOKENIZER="tokenizer/mix32k.model"
BLOCK=1024
random.seed(42)

MIX={
"raw_data/fineweb_edu.txt":0.35,
"raw_data/wikipedia.txt":0.20,
"raw_data/openhermes.txt":0.15,
"raw_data/alpaca_gpt4.txt":0.12,
"raw_data/books.txt":0.08,
"raw_data/codealpaca.txt":0.06,
"raw_data/gsm8k.txt":0.04,
}

sp=spm.SentencePieceProcessor(model_file=TOKENIZER)

os.makedirs("datasets",exist_ok=True)

train_file=open("datasets/train_v5.bin","wb")
val_file=open("datasets/val_v5.bin","wb")

chunk_count=0
token_count=0

for path,ratio in MIX.items():

    if not os.path.exists(path): continue

    print("loading",path)

    with open(path,encoding="utf8",errors="ignore") as f:
        text=f.read()

    docs=[d.strip() for d in text.split("\n\n") if len(d)>30]

    target=int(len(docs)*ratio*10)

    sample=random.sample(docs,min(target,len(docs)))

    for d in sample:

        tok=sp.encode(d)

        for i in range(0,len(tok),BLOCK):

            c=tok[i:i+BLOCK+1]

            if len(c)<32:
                continue

            arr=np.array(c,dtype=np.uint16)

            if random.random()<0.1:
                arr.tofile(val_file)
            else:
                arr.tofile(train_file)

            chunk_count+=1
            token_count+=len(c)

            if chunk_count%50000==0:
                print("chunks",chunk_count,"tokens",token_count)

train_file.close()
val_file.close()

print("DONE")
print("chunks:",chunk_count)
print("tokens:",token_count)
