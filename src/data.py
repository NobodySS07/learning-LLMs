import torch
from torch.utils.data import Dataset, DataLoader
from src.utils import normalize_review

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # 1. Fetch the text and label (it is ALREADY normalized from the notebook)
        clean_text = self.texts[idx]
        label = self.labels[idx]
        
        # 2. Tokenize
        token_ids = self.tokenizer.encode(clean_text, max_length=self.max_length)
        
        # 3. Convert to Tensors
        token_tensor = torch.tensor(token_ids, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return token_tensor, label_tensor

def create_dataloaders(train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, tokenizer, batch_size=32, max_length=256):
        
    # 1. Instantiate the Datasets
    train_dataset = IMDBDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = IMDBDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = IMDBDataset(test_texts, test_labels, tokenizer, max_length)

    # 2. Create the DataLoaders
    # We shuffle the training data to prevent the model from learning sequence patterns
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # We do not need to shuffle validation and test data
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader