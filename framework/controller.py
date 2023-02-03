import logging
import os
import torch


class Saver(object):
    def __init__(self, model_path, fn):
        self.model_path = model_path
        self.fn = fn

    def save(self, save_dict):
        torch.save(save_dict, os.path.join(self.model_path, self.fn))

    def load(self):
        return torch.load(os.path.join(self.model_path, self.fn))


class EarlyStopper(object):
    def __init__(self, logger, small_is_better, patience, model_path, fn):
        self.small = small_is_better
        if self.small:
            self.best_loss = 100000.0
        else:
            self.best_loss = -100000.0
        self.logger = logger
        self.saver = Saver(model_path, fn)
        self.num_epoch_no_improvement = 0
        self.patience = patience

    def step_small(self, valid_loss, save_dict):
        if valid_loss < self.best_loss:
            self.logger.info("Validation loss decreases from {:.4f} to {:.4f}".format(self.best_loss, valid_loss))
            self.best_loss = valid_loss
            self.num_epoch_no_improvement = 0
            self.saver.save(save_dict)
        else:
            self.logger.info(
                "Validation loss does not decrease from {:.4f}, num_epoch_no_improvement {}".format(self.best_loss,
                                                                                                    self.num_epoch_no_improvement))
            self.num_epoch_no_improvement += 1
        if self.num_epoch_no_improvement == self.patience:
            self.logger.info("Early Stopping")
            exit()

    def step_big(self, valid_loss, save_dict):
        if valid_loss > self.best_loss:
            self.logger.info("Validation loss increases from {:.4f} to {:.4f}".format(self.best_loss, valid_loss))
            self.best_loss = valid_loss
            self.num_epoch_no_improvement = 0
            self.saver.save(save_dict)
        else:
            self.logger.info(
                "Validation loss does not increase from {:.4f}, num_epoch_no_improvement {}".format(self.best_loss,
                                                                                                    self.num_epoch_no_improvement))
            self.num_epoch_no_improvement += 1
        if self.num_epoch_no_improvement == self.patience:
            self.logger.info("Early Stopping")
            exit()

    def step(self, valid_loss, save_dict):
        if self.small:
            self.step_small(valid_loss, save_dict)
        else:
            self.step_big(valid_loss, save_dict)
