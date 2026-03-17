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

train=open("datasets/train_v5.bin","wb")
val=open("datasets/val_v5.bin","wb")

chunks=0
tokens=0

BATCH=1024

for path,ratio in MIX.items():

    if not os.path.exists(path): 
        continue

    print("loading",path)

    txt=open(path,encoding="utf8",errors="ignore").read()

    docs=[d.strip() for d in txt.split("\n\n") if len(d)>30]

    target=int(len(docs)*ratio*10)

    sample=random.sample(docs,min(target,len(docs)))

    for i in range(0,len(sample),BATCH):

        batch=sample[i:i+BATCH]

        tok_batch=sp.encode(batch)

        for tok in tok_batch:

            for j in range(0,len(tok),BLOCK):

                c=tok[j:j+BLOCK+1]

                if len(c)<32:
                    continue

                arr=np.array(c,dtype=np.uint16)

                if random.random()<0.1:
                    arr.tofile(val)
                else:
                    arr.tofile(train)

                chunks+=1
                tokens+=len(c)

                if chunks%50000==0:
                    print("chunks:",chunks,"tokens:",tokens)

train.close()
val.close()

print("DONE")
print("chunks:",chunks)
print("tokens:",tokens)
