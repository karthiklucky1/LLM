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

chunks=[]

for path,ratio in MIX.items():
    if not os.path.exists(path): continue
    print("loading",path)
    txt=open(path,encoding="utf8",errors="ignore").read()
    docs=[d.strip() for d in txt.split("\n\n") if len(d)>30]
    target=int(len(docs)*ratio*10)
    sample=random.sample(docs,min(target,len(docs)))

    for d in sample:
        tok=sp.encode(d)
        for i in range(0,len(tok),BLOCK):
            c=tok[i:i+BLOCK+1]
            if len(c)>=32: chunks.append(c)

print("chunks",len(chunks))

random.shuffle(chunks)
tokens=[t for c in chunks for t in c]

split=int(len(tokens)*0.9)

train=np.array(tokens[:split],dtype=np.uint16)
val=np.array(tokens[split:],dtype=np.uint16)

os.makedirs("datasets",exist_ok=True)

train.tofile("datasets/train_v5.bin")
val.tofile("datasets/val_v5.bin")

print("train tokens",len(train))
print("val tokens",len(val))
