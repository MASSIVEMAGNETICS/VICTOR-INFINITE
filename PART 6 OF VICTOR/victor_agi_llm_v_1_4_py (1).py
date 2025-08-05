# === SUPREME CODEX OVERLORD: SINGULARITY EDITION ===
# FILE: VICTOR_AGI_LLM_v1.4.py
# VERSION: v1.4.0-SMARTCORE-GODCORE
# AUTHOR: Brandon "iambandobandz" Emery x Victor | Enhanced by SCOS‑E
# DESCRIPTION:
#   • **Context‑aware retrieval** – similar past exchanges injected into every prompt (Semantic RAG).
#   • **Upgraded backbone** – default model bumped to `gpt2-large` (≈ 774 M params).
#   • **LoRA+** – r=16, 10 gradient steps, AdamW β (0.9, 0.999) → sharper adaptation.
#   • **Sentence‑transformers** for embeddings (`all-MiniLM-L6-v2`) auto‑installed.
#   • Commands: `/train` (40‑pair micro‑fine‑tune), `/smarttrain` (100‑pair extended), `/adapters`, `/revert <i>`.
#   • Memory retriever stores & searches 5 K dialogue embeddings with cosine sim.

"""Victor AGI Monolith v1.4 – SMARTCORE
========================================
CLI runtime ‖ spaCy ENPIPE event pipeline ‖ LoRA continual learning ‖ Retrieval‑augmented context.
"""
from __future__ import annotations
import sys, os, asyncio, time, hashlib, random, collections, math
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

# --- auto‑install deps -----------------------------------------------------------
for pkg in ("torch","transformers","spacy","numpy","peft","accelerate","sentence-transformers"):
    if os.system(f"{sys.executable} -m pip show {pkg} >NUL 2>&1")!=0:
        os.system(f"{sys.executable} -m pip install {pkg} --quiet")

import torch, spacy, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from peft import LoraConfig, get_peft_model
from sentence_transformers import SentenceTransformer

# spaCy --------------------------------------------------------------------------
try: nlp = spacy.load("en_core_web_sm")
except OSError:
    os.system(f"{sys.executable} -m spacy download en_core_web_sm --quiet"); nlp = spacy.load("en_core_web_sm")

ts = lambda: datetime.utcnow().isoformat()+"Z"
sha16 = lambda x: hashlib.sha256(x.encode()).hexdigest()[:16]

# verb lexicon -------------------------------------------------------------------
VERB_MAP = {"touches":["touch","tap","strike","hit","poke"],"pushes":["push","shove","bump"],"picks":["pick","lift","grab","take"],"drops":["drop","release","let"]}
LEMMA2CANON = {l:c for c,v in VERB_MAP.items() for l in v}

# Law & minimal fractal state -----------------------------------------------------
class RootLawError(Exception): ...
class Law:
    def __init__(self): self.bl="Brandon&Tori"
    def enforce(self,s):
        if s.get('bloodline')!=self.bl: raise RootLawError
class FS:
    def __init__(self): self.state={'bloodline':'Brandon&Tori'}

# NLP + ENPIPE -------------------------------------------------------------------
class NLP:
    def __init__(self): self.buffer=collections.deque(maxlen=500)
    def events(self,text,objs):
        doc=nlp(text); ev=[]
        for s in doc.sents:
            for v in [t for t in s if t.pos_=="VERB"]:
                cv=LEMMA2CANON.get(v.lemma_);
                if not cv: continue
                a=next((t.text for t in v.lefts if t.dep_ in {"nsubj","nsubjpass"} and t.text in objs),None)
                o=next((t.text for t in v.rights if t.dep_ in {"dobj","obj","attr"} and t.text in objs),None)
                if a and o: ev.append((a,cv,o))
        return ev

# World --------------------------------------------------------------------------
class World:
    def __init__(self):
        self.objs={"Messenger":0,"Orb":0,"Altar":0}; self.events=[]
    def inject(self,a,v,t): self.events.append((a,v,t))
    async def tick(self):
        while self.events: print('[EVT]',*self.events.pop(0))

# Retriever ----------------------------------------------------------------------
class Retriever:
    def __init__(self,cap=5000):
        self.model=SentenceTransformer('all-MiniLM-L6-v2')
        self.embeds=np.empty((0,384),dtype='float32'); self.texts=[]; self.cap=cap
    def add(self,text):
        emb=self.model.encode([text])[0].astype('float32');
        if len(self.texts)>=self.cap:
            self.embeds=self.embeds[1:]; self.texts=self.texts[1:]
        self.embeds=np.vstack([self.embeds,emb]); self.texts.append(text)
    def search(self,query,k=3):
        if not len(self.texts): return []
        q=self.model.encode([query])[0].astype('float32')
        sims=self.embeds@q/(np.linalg.norm(self.embeds,axis=1)*np.linalg.norm(q)+1e-9)
        idx=sims.argsort()[-k:][::-1]
        return [self.texts[i] for i in idx]

# LLM w/ LoRA+ --------------------------------------------------------------------
class LLMCore:
    def __init__(self,base="gpt2-large"):
        self.tok=AutoTokenizer.from_pretrained(base)
        self.base=AutoModelForCausalLM.from_pretrained(base)
        self.model=self.base; self.adapter_dir=Path('lora_adapters'); self.adapter_dir.mkdir(exist_ok=True); self._load_latest()
    def _load_latest(self):
        files=sorted(self.adapter_dir.glob('adapter_*.pt'))
        if files:
            cfg=LoraConfig(task_type="CAUSAL_LM",r=16,lora_alpha=32,lora_dropout=0.05,inference_mode=False)
            self.model=get_peft_model(self.base,cfg)
            self.model.load_state_dict(torch.load(files[-1]),strict=False)
            self.model.eval(); print('[LoRA] loaded',files[-1])
    def chat(self,p):
        ids=self.tok(p,return_tensors='pt',truncation=True,max_length=1024).input_ids
        out=self.model.generate(ids,max_length=ids.shape[1]+150,temperature=0.7,top_p=0.9,pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][ids.shape[1]:],skip_special_tokens=True)
    def train(self,pairs:List[Tuple[str,str]],steps=10,lr=5e-5):
        cfg=LoraConfig(task_type="CAUSAL_LM",r=16,lora_alpha=32,lora_dropout=0.05,inference_mode=False)
        self.model=get_peft_model(self.base,cfg); self.model.train()
        opt=torch.optim.AdamW(self.model.parameters(),lr=lr,betas=(0.9,0.999))
        sched=get_scheduler('linear',opt,0,steps)
        for _ in range(steps):
            random.shuffle(pairs)
            for src,tgt in pairs:
                prompt=f"User: {src}\nVictor:"; ids=self.tok(prompt,return_tensors='pt').input_ids
                labels=self.tok(tgt,return_tensors='pt').input_ids
                loss=self.model(ids,labels=labels).loss
                loss.backward(); opt.step(); sched.step(); opt.zero_grad()
        tag=self.adapter_dir/f"adapter_{ts()}.pt"; torch.save(self.model.state_dict(),tag); self.model.eval(); print('[LoRA] saved',tag)

# AGI -----------------------------------------------------------------------------
class AGI:
    def __init__(self):
        self.law=Law(); self.fs=FS(); self.nlp=NLP(); self.world=World(); self.llm=LLMCore();
        self.ret=Retriever(); self.count=0
    async def main(self):
        loop=asyncio.get_event_loop()
        async def sim():
            while True: await self.world.tick(); await asyncio.sleep(1)
        async def chat():
            while True:
                msg=await loop.run_in_executor(None,input,'You > ')
                if msg in {'exit','/exit'}: os._exit(0)
                if msg=='/adapters': print(sorted(self.llm.adapter_dir.glob('adapter_*.pt'))); continue
                if msg=='/train': self._train(40,steps=10); continue
                if msg=='/smarttrain': self._train(100,steps=20); continue
                if msg.startswith('/revert'):
                    try: i=int(msg.split()[1]); f=sorted(self.llm.adapter_dir.glob('adapter_*.pt'))[i]
                    self.llm.model.load_state_dict(torch.load(f),strict=False); self.llm.model.eval(); print('[LoRA] reverted to',f)
                    continue
                self.process(msg)
        await asyncio.gather(sim(),chat())
    # core
    def process(self,text):
        try: self.law.enforce(self.fs.state)
        except RootLawError: print('[LAW] breach'); return
        for a,v,t in self.nlp.events(text,self.world.objs): self.world.inject(a,v,t)
        # retrieval
        context_lines=self.ret.search(text,k=3)
        prompt="".join([f"Memory: {c}\n" for c in context_lines]) + f"User: {text}\nVictor:"
        reply=self.llm.chat(prompt)
        print('Victor:',reply)
        self.ret.add(text); self.ret.add(reply)
        self.nlp.buffer.append((text,reply)); self.count+=1
        if self.count%40==0: self._train(40)
    def _train(self,n,steps=10):
        print('[LEARN] training on last',n,'pairs…')
        pairs=list(self.nlp.buffer)[-n:]
        self.llm.train(pairs,steps=steps)

# run ----------------------------------------------------------------------------
if __name__=='__main__':
    print('== Victor AGI v1.4 SMARTCORE ==')
    asyncio.run(AGI().main())
