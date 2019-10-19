import torch
from torch.autograd import Variable


class WhaleTrainer:
    """
    """
    losses = {}
    accuracies = {}

    def __init__(self, model, train_dataloader, val_dataloader, num_epochs,
                Loss, optimizer, model_path=None,save_path=None, save=False, load=False):
        self.model = model
        self.save_choice = save
        if load:
           self.model.load_state_dict(torch.load(model_path))
           self.model.train()
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.save = save
        self.Loss = Loss
        self.save_path = save_path
        self.num_epochs = num_epochs
        self.optimizer = optimizer

    def train(self):
        print('begginning to train the machine')
        count = 0
        correct = 0
        total = 0
        for epoch in range(self.num_epochs):
            self.losses[epoch] = []
            self.accuracies[epoch] = []
            for i, sample_data in enumerate(self.train_dataloader, 0):
                count +=1
                batch = sample_data['image'].to(torch.float32)
                label = sample_data['label']
                batch = Variable(batch)
                self.optimizer.zero_grad()
                output = self.model(batch)
                loss = self.Loss(output, label)
                self.losses[epoch].append(loss)
                loss.backward()
                self.optimizer.step()
                total += batch.size(0)
                _, predicted = torch.max(output, 1)
                correct += (predicted == label).sum().item()
                accuracy = correct/total
                self.accuracies[epoch].append(accuracy)
                if count % 10 == 0:
                    print(f'loss at iteration {i}: {loss}')
                    print(f'accuracy: {accuracy}')



