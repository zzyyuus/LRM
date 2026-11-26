import torch
def targeted_mix(x,y,harbor,water,rng,probability=.25,alpha=.2):
    # Pair only original, same-batch training images; never recursively mix.
    labels=y.cpu().tolist();partner=list(range(len(labels)));lam=[1.]*len(labels)
    ids={k:[i for i,v in enumerate(labels) if v==k] for k in (harbor,water)}
    eligible=mixed=0
    for i,label in enumerate(labels):
        if label not in ids:continue
        opposite=water if label==harbor else harbor
        if not ids[opposite]:continue
        eligible+=1
        if rng.random()<probability:
            partner[i]=rng.choice(ids[opposite]);lam[i]=rng.betavariate(alpha,alpha);mixed+=1
    p=torch.tensor(partner,device=x.device);w=torch.tensor(lam,device=x.device,dtype=x.dtype)
    return w[:,None,None,None]*x+(1-w[:,None,None,None])*x[p],y[p],w,eligible,mixed

def pair_mix(x,y,class_a,class_b,rng,probability=.25,alpha=.2):
    """定向配对混合的通用版：把两类（如 harbor/water、harbor/river）互相配对。

    与 targeted_mix 完全同构，只是类别可配置；默认参数下两者行为逐位一致。
    其余类别原样通过（lam=1，partner 为自己）。
    """
    return targeted_mix(x,y,class_a,class_b,rng,probability,alpha)

def mixup_batch(x,y,rng,probability=.5,alpha=.2):
    """标准（全类）Mixup：每张图以 probability 的概率与同批次中另一张随机图混合。

    与定向版不同，这里不挑类别、也不要求批次里同时出现某两类，因此可以直接
    用在没有 harbor/water 的数据集（如 UCMerced）上，用于"两个数据集统一配方"的对照。
    每张图的混合系数 λ ~ Beta(alpha, alpha)，标签按同一 λ 混合（软标签）。
    """
    labels=y.cpu().tolist();n=len(labels)
    partner=list(range(n));lam=[1.]*n
    eligible=mixed=0
    if n<2:
        p=torch.arange(n,device=x.device)
        return x,y,torch.ones(n,device=x.device,dtype=x.dtype),0,0
    for i in range(n):
        eligible+=1
        if rng.random()<probability:
            j=rng.randrange(n)
            while j==i:
                j=rng.randrange(n)
            partner[i]=j;lam[i]=rng.betavariate(alpha,alpha);mixed+=1
    p=torch.tensor(partner,device=x.device);w=torch.tensor(lam,device=x.device,dtype=x.dtype)
    return w[:,None,None,None]*x+(1-w[:,None,None,None])*x[p],y[p],w,eligible,mixed
