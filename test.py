import torch

t1=torch.tensor([[0,0,1]])
t2=t1.view(1,1,-1)
print(t2)