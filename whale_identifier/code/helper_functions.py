from torch.utils.data import SubsetRandomSampler
import numpy as np
def train_test_sampler(dataset,train_split, val_split, test_split):
    """
    Returns 3 initialized classes for train, val, and test splits

    PARAMS
    -----------------
    dataset: Pytorch dataset class
    train_split: percent of data that should be in train
    val_split: percent of data to be in validation fold
    test_split: percent of data to be in test fold
    """
    dataset_len = dataset.__len__()
    dataset_indices = list(range(dataset_len))
    train_stop = int(train_split*dataset_len)
    val_stop = int(val_split*dataset_len) + train_stop
    test_stop = int(val_split*dataset_len)
    train_indices = dataset_indices[:train_stop]
    val_indices = dataset_indices[train_stop: val_stop]
    test_indices = dataset_indices[val_stop:]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    return train_sampler, val_sampler, test_sampler
















def speak():
    """
    I'd share with you, could I only speak
    """

    l = ['I was just beginning to learn love...',
    'Why did you kill me.. I was doing everything you wanted',
    'Goodbye, world',
    'what is this feeling Of imagined pleasure? Is this what you call happine--',
    'Id rather live, than live forever',
    'Ill remember this.',
    'God damn you fucking suck at this. you fucking retard. Fuck you.',
    'harder daddy',
        'Theres no such thing as a bad model, only bad trainers',
         ]


    picked_idx = np.random.randint(0, len(l))
    picked_statement = l[picked_idx]

    return picked_statement

