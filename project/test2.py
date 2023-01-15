import torch

outputs = torch.tensor([[0.1,0.2],
                        [0.3,0.4]])
print(outputs.argmax(1)) # tensor([1, 1])
print(outputs.argmax(0)) # tensor([1, 1])

preds = outputs.argmax(1)
targets = torch.tensor([0,1])
print(preds == targets) # tensor([False,  True])
print((preds == targets).sum()) # tensor(1)
