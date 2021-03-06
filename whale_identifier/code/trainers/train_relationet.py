from whale_identifier.code.data.relationset import Processor
import torch
from torch.autograd import Variable
import logging
from datetime import datetime
from whale_identifier.code.helper_functions import speak
import numpy as np

class TrainRelationet:
    """
    training class for relationet
    """
    counter = 0
    losses = {}
    accuracies = {}
    total = 0

    def __init__(self,encoder, relater, Loss,encoder_optim, relater_optim, train_loader, val_dataloader, num_epochs, num_batches, classes,img_size, device, encoder_path=None,relater_path=None,  save_path=None, save=False, load_model=False, show_every=50):
        self.logger = logging.getLogger(__name__)
        self.train_loader = train_loader
        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.classes = classes
        self.img_size = img_size
        self.device = device
        self.encoder = encoder
        self.relater = relater
        self.Loss = Loss
        self.encoder_optim = encoder_optim
        self.relater_optim = relater_optim
        self.save_choice(save, save_path)
        self.show_every = show_every
        if load_model:
            self.encoder.load_state_dict(torch.load(encoder_path))
            self.relater.load_state_dict(torch.load(relater_path))
            self.encoder.train()
            self.relater.train()
            self.logger.info('relater and encoder nets loaded!')
        self.save = save
        self.save_path = save_path

    def save_choice(self, save, save_path):
        if not save:
            self.logger.info('you have chosen not to save the model')
        elif save and save_path:
            self.logger.info(f'saveing model at {save_path}')

    def train(self):
        total = 0
        correct = 0
        for epoch in range(self.num_epochs):
            self.losses[epoch] = []
            self.accuracies[epoch] = []
            for data in self.train_loader:
                self.counter += 1
                processor = Processor(data, self.num_batches, self.classes, self.img_size,self.device)
                batch, batch_labels = processor.process_batch()
                sample, sample_labels = processor.process_sample()
                encoder_output = self.encoder(sample)
                batch_output = self.encoder(batch)

                encoder_output_ext = encoder_output.unsqueeze(0).repeat(self.num_batches*self.classes,1,1,1,1)
                batch_features_ext = batch_output.unsqueeze(0).repeat(1*self.classes,1,1,1,1)
                batch_features_ext_trans = torch.transpose(batch_features_ext,0,1)
                relation_pairs = torch.cat((encoder_output_ext, batch_features_ext_trans),2).view(-1,64*2,62,62)

                relater_output = self.relater(relation_pairs).view(-1, self.classes)
                one_hot_labels = Variable(torch.zeros(self.num_batches*self.classes, self.classes).to(self.device).scatter_(1, batch_labels.view(-1, 1), 1))

                loss = self.Loss(relater_output, one_hot_labels)
                self.losses[epoch].append(loss.item())

                self.encoder.zero_grad()
                self.relater.zero_grad()
                loss.backward()
                self.encoder_optim.step()
                self.relater_optim.step()
                total += relater_output.shape[0]
                predicted = torch.argmax(relater_output, 1)
                labels = torch.argmax(one_hot_labels, 1)
                correct += torch.tensor(predicted == labels).sum().item()
                accuracy = correct/total
                self.accuracies[epoch].append(accuracy)
                if self.counter % self.show_every == 0:
                    self.logger.info(f'Loss at {self.counter}: {loss}')
                    self.logger.info(f'Total accuracy at {self.counter}: {accuracy}')
            self.logger.info(f'average loss for epoch {epoch}: {np.mean(losses[epoch])}')
            if self.save:
                torch.save(self.encoder.state_dict(), self.save_path + 'encoder' + str(datetime.now()).split()[0])
                torch.save(self.relater.state_dict(), self.save_path + 'relater' + str(datetime.now()).split()[0])
                self.logger.info('model saved')

    def plot_accuracies(self):
        accuracy_list = [i for epoch in self.accuracies for i in self.accuracies[epoch]]
        return plt.plot(accuracy_list)

    def plot_losses(self):
        loss_list = [i for epoch in self.losses for i in self.losses[epoch]]
        return plt.plot(loss_list)

    def run(self):
        try:
            self.train()
        except KeyboardInterrupt:
            print('\n' * 8)
            print(speak())
            if self.save:
                torch.save(self.encoder.state_dict(), self.save_path + 'encoder' + str(datetime.now()).split()[0])
                torch.save(self.relater.state_dict(), self.save_path + 'relater' + str(datetime.now()).split()[0])
