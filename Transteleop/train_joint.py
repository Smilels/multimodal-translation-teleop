import time
import copy
import os
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import numpy as np
from tensorboardX import SummaryWriter

if __name__ == '__main__':
    opt = TrainOptions().parse()  # get training options

    logger = SummaryWriter(os.path.join(opt.log_path, opt.name))

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    print('The number of training images = %d' % len(dataset))
    opt_test = copy.deepcopy(opt)
    opt_test.phase = 'test'
    opt_test.isTrain = False
    dataset_test = create_dataset(opt_test)  # create a dataset given opt.dataset_mode and other options
    print("test data number is: ", len(dataset_test.dataset))

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    total_iters = 0  # the total number of training iterations

    thresh_acc = [0.2, 0.25, 0.3]

    for epoch in range(opt.epoch_count,
                       opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        correct_shadow = [0, 0, 0]
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        # train
        model.train()
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters(epoch)  # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / len(dataset), losses)

            if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (
            epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()  # update learning rates at the end of every epoch.

        # eval
        model.eval()
        for i, data in enumerate(dataset_test):
            model.set_input(data)  # unpack data from data loader
            model.test()  # run inference
            joint_acc_error = model.get_current_error()
            # compute acc
            res_shadow = [np.sum(np.sum(abs(joint_acc_error.cpu().data.numpy()) < thresh,
                                        axis=-1) == 22) for thresh in thresh_acc]
            correct_shadow = [c + r for c, r in zip(correct_shadow, res_shadow)]

        acc_shadow = [float(c) / float(len(dataset_test.dataset)) for c in correct_shadow]
        logger.add_scalar('test_acc_shadow0.2', acc_shadow[0], epoch)
        logger.add_scalar('test_acc_shadow0.25', acc_shadow[1], epoch)
        logger.add_scalar('test_acc_shadow0.3', acc_shadow[2], epoch)
