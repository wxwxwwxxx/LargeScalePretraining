import logging
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from framework.utils import iou, dice_coef, binary_mean_iou_eval


class CustomLogger():
    def __init__(self, logger_name, conf, tboard_img_num=16):
        # logger
        self.conf = conf
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        fh = logging.FileHandler(os.path.join(conf.logs_path, "output.log"))
        fh.setLevel(logging.INFO)
        ch_formatter = logging.Formatter('%(asctime)s,%(message)s', datefmt='%H:%M:%S')
        fh_formatter = logging.Formatter('%(asctime)s,[%(name)s],%(message)s', datefmt='%Y/%m/%d %H:%M:%S')
        ch.setFormatter(ch_formatter)
        fh.setFormatter(fh_formatter)
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)
        # tboard
        self.writer = SummaryWriter(self.conf.tboard_path)
        self.tboard_img_num = tboard_img_num
        # metrics
        self.train_losses = []
        self.train_image_dict = None
        self.valid_image_dict = None
        self.valid_ious = []
        self.valid_dices = []
        self.valid_losses = []

    def get_logger(self):
        return self.logger

    def get_tboard_writer(self):
        return self.writer

    def train_iter_record_seg(self, loss, img_dict):
        self.train_losses.append(loss)
        if self.train_image_dict is None:
            self.train_image_dict = img_dict[-self.tboard_img_num:]
        else:
            for i in img_dict:
                self.train_image_dict[i] = torch.cat([self.train_image_dict[i], img_dict[i]], 0)[-self.tboard_img_num:]

    def train_write_to_log_seg(self, epoch):
        self.logger.debug('Epoch [{}/{}], training loss: {:.6f}'.format(epoch + 1, self.conf.nb_epoch,
                                                                             np.average(self.train_losses)))
        for i in self.train_image_dict:
            img_shape = list(self.train_image_dict[i])
            assert len(img_shape) == 4 or len(img_shape) == 5
            if len(img_shape) == 5:
                self.train_image_dict[i] = self.train_image_dict[i][:, :, :, :, img_shape[-1] // 2]
            self.writer.add_images(f"Image/train_{i}", self.train_image_dict[i], epoch)
        self.writer.add_scalar('Loss/train', np.average(self.train_losses), epoch)
        self.train_losses = []
        self.train_image_dict = None

    def test_iter_record_seg(self, loss, img_dict):
        # loss can be calculated during main process, so we take loss as a argument
        iou_metric = binary_mean_iou_eval(img_dict['gt'], img_dict['pred']).item()
        dice_metric = dice_coef(img_dict['gt'], img_dict['pred']).item()

        self.valid_losses.append(loss)
        self.valid_ious.append(iou_metric)
        self.valid_dices.append(dice_metric)

        if self.valid_image_dict is None:
            self.valid_image_dict = img_dict[-self.tboard_img_num:]
        else:
            for i in img_dict:
                self.valid_image_dict[i] = torch.cat([self.valid_image_dict[i], img_dict[i]], 0)[-self.tboard_img_num:]

    def test_write_to_log_seg(self, epoch):
        valid_iou, valid_dice, valid_loss = np.average(np.valid_ious), np.average(np.valid_dices), np.average(np.valid_losses)
        self.writer.add_scalar('Loss/valid', np.average(np.valid_losses), epoch)
        self.writer.add_scalar('Loss/valid_iou', np.average(np.valid_ious), epoch)
        self.writer.add_scalar('Loss/valid_dice', np.average(np.valid_dices), epoch)
        self.logger.info(
            "Epoch {}, validation iou is {:.4f}, validation dice is {:.4f}, validation loss is {:.4f}".format(
                epoch + 1, valid_iou, valid_dice, valid_loss))
        for i in self.train_image_dict:
            img_shape = list(self.train_image_dict[i])
            assert len(img_shape) == 4 or len(img_shape) == 5
            if len(img_shape) == 5:
                self.train_image_dict[i] = self.train_image_dict[i][:, :, :, :, img_shape[-1] // 2]
            self.writer.add_images(f"Image/valid_{i}", self.valid_image_dict[i], epoch)
