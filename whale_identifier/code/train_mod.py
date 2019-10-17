

def TrainModel():
    """
    train model

    """
    for epoch in num_epochs:
        for data_sample in dataloader:
            batch = data_sample['image']
            label = data_sample['label']
