from torch.utils.data import Dataset

class Loader(Dataset):
    def __init__(self, tokens,masks,label):
        self.tokens=tokens
        self.masks=masks
        self.label=label
        
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, index):
        return (self.tokens[index],self.masks[index],self.label[index])