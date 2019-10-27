from whale_identifier.code.data.relationset import Processor
import torch
from torch.autograd import Variable


class TrainRelationet:
    """
    training class for relationet
    """
    counter = 0
    losses = {}
    accuracies = {}

    def __init__(self,encoder, relater, Loss,encoder_optim, relater_optim, train_loader, num_epochs, num_batches, classes,img_size, device):
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

    def run(self):
        for epoch in range(self.num_epochs):
            self.losses[epoch] = []
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
                self.losses[epoch].append(loss)

                self.encoder.zero_grad()
                self.relater.zero_grad()
                loss.backward()
                self.encoder_optim.step()
                self.relater_optim.step()
                if self.counter % 1 == 0:
                    print(f'Loss at {self.counter}: {loss}')

