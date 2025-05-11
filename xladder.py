import torch.nn as nn
import pickle
import torch.nn.functional as F
import torch

# torch.autograd.set_detect_anomaly(True)
detladder = None
detladders = []
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def loadMatrices(fnom):
    print("loading ladder",fnom)
    mlps, wq, wk, wv = [], [], [], []
    try:
        with open(fnom,"rb") as f:
            dproj=pickle.load(f)
            uproj=pickle.load(f)
            while True:
                mlps.append(pickle.load(f))
                wq.append(pickle.load(f))
                wk.append(pickle.load(f))
                wv.append(pickle.load(f))
    except:
        pass
    assert len(mlps)==len(wq)
    assert len(mlps)==len(wk)
    assert len(mlps)==len(wv)
    dproj = dproj.to(dev)
    uproj = uproj.to(dev)
    for i in range(len(mlps)):
        mlps[i] = mlps[i].to(dev)
        wq[i] = wq[i].to(dev)
        wk[i] = wk[i].to(dev)
        wv[i] = wv[i].to(dev)

    return dproj, uproj, mlps, wq, wk, wv
 
class Ladder(nn.Module):
    def __init__(self):
        super().__init__()
        self.debug = False
        self.lay2keep = (10,15)
        self.activs = []

    def createRandom(self,h0,h1,nl):
        self.h1=h1
        self.dproj = nn.Linear(h0,h1)
        self.uproj = nn.Linear(h1,h0)
        mlps, wq, wk, wv = [], [], [], []
        for i in range(nl):
            mlps.append(nn.Sequential(nn.Linear(h1,4*h1),nn.ReLU(),nn.Linear(4*h1,h1)))
            wq.append(nn.Linear(h1,h1))
            wk.append(nn.Linear(h1,h1))
            wv.append(nn.Linear(h1,h1))
        self.mlps = nn.ModuleList(mlps)
        self.wq = nn.ModuleList(wq)
        self.wk = nn.ModuleList(wk)
        self.wv = nn.ModuleList(wv)
        with torch.no_grad():
            # init conservatrice
            self.uproj.weight.zero_()
            self.uproj.bias.zero_()
 
    def detsave(self, fnom):
        with open(fnom,"wb") as f:
            pickle.dump(self.dproj.to("cpu"),f)
            pickle.dump(self.uproj.to("cpu"),f)
            self.dproj.to(dev)
            self.uproj.to(dev)
            for i in range(len(self.mlps)):
                pickle.dump(self.mlps[i].to("cpu"),f)
                pickle.dump(self.wq[i].to("cpu"),f)
                pickle.dump(self.wk[i].to("cpu"),f)
                pickle.dump(self.wv[i].to("cpu"),f)
                self.mlps[i].to(dev)
                self.wq[i].to(dev)
                self.wk[i].to(dev)
                self.wv[i].to(dev)
 
    def detload(self, fnom):
        self.dproj, self.uproj, self.mlps, self.wq, self.wk, self.wv = loadMatrices(fnom)

    def ladsatt(self, i, x):
        Q = self.wq[i](x)  # (batch_size, seq_len, embed_dim)
        K = self.wk[i](x)  # (batch_size, seq_len, embed_dim)
        V = self.wv[i](x)  # (batch_size, seq_len, embed_dim)

        seq_len = x.size(1)
        if self.testtime:
            if seq_len == 1:
                # assume that seqlen>1 is eq. to start of sentence
                K = torch.cat([self.kvcache[i]['k'], K], dim=1)
                V = torch.cat([self.kvcache[i]['v'], V], dim=1)
            elif i==0: self.kvcache = [None]*len(self.mlps)
            self.kvcache[i] = {'k': K.detach(), 'v': V.detach()}

        attn_scores = Q @ K.transpose(-2, -1) / (self.h1 ** 0.5)  # (batch_size, seq_len, seq_len)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()  # (seq_len, seq_len)
        attn_scores = attn_scores.masked_fill(~causal_mask, -1e9)
        if detladders[0].attmask is not None:
            # when batchsize > 1
            mask = detladders[0].attmask.unsqueeze(1)  # (batch_size, 1, seq_len)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, seq_len, seq_len)
        satt = attn_weights @ V  # (batch_size, seq_len, embed_dim) 
        return satt

    def forward(self, z):
        rs = self.dproj(self.activs[0]) # (batch_size, seq_len, h1)
        rs = rs+self.ladsatt(0,rs)
        rs = rs+self.mlps[0](rs)

        rs = rs+self.dproj(self.activs[1])
        rs = rs+self.ladsatt(1,rs)
        rs = rs+self.mlps[1](rs)

        for i in range(2,len(self.mlps)):
            rs = rs+self.ladsatt(i,rs)
            rs = rs+self.mlps[i](rs)

        # merging and unembedding
        x99 = z
        z = self.uproj(rs) + x99
        if self.debug: z = x99
        self.activs = []

        # apres le 1er forward, on ne mask plus car le KV-cache ne donne que 1 token
        # the caller must setup this mask before processing a sentence when batchsize>1 !
        detladders[0].attmask = None
        return z

def myhook(layer, input, output):
    global detladder
    z = output[0]
    detladder.activs.append(z.detach().to(torch.float32))

def myhookfin(layer, input, output):
    global detladder
    z = output[0]
    zt = z.dtype
    z = z.to(torch.float32)
    z = detladder.forward(z)
    z = z.to(zt)
    return (z,)

def getParmNorm():
    global detladder
    n=sum([p.norm().item() for p in detladder.parameters()])
    return n

def debugmode(noladder=True):
    global detladder
    detladder.debug=noladder

def _addLadder(mod):
    global detladder, detladders
    detladder = Ladder()
    detladder.attmask = None # dont know yet the attmask: dont forget to set it before calling mod.forward() !!
    detladders.append(detladder)
    if len(detladders)==1:
        # first ladder added to the model
        for p in mod.parameters(): p.requires_grad=False
        mod.add_module("ladder",detladder)
        for i in detladder.lay2keep:
            mod.model.layers[i].detlayer = i
            mod.model.layers[i].register_forward_hook(myhook)
        mod.model.layers[-1].register_forward_hook(myhookfin)

def addLadder(mod,h1,nl):
    global detladder, detladders
    _addLadder(mod)
    detladder.createRandom(mod.config.hidden_size,h1,nl)
    detladder.nom = "_"+str(h1)+"_"+str(nl) 
    detladder.testtime = False
    detladder.to(dev)
    print("added ladder size",sum([p.numel() for p in detladder.parameters()]))

def loadandAddLadder(mod,fich):
    global detladder, detladders
    _addLadder(mod)
    detladder.dproj, detladder.uproj, detladder.mlps, detladder.wq, detladder.wk, detladder.wv = loadMatrices(fich)
    detladder.h1 = detladder.dproj.weight.shape[0]
    detladder.nom = fich 
    detladder.testtime = True
    detladder.to(dev)
    print("loaded ladder size",sum([p.numel() for p in detladder.parameters()]))

