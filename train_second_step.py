"""
@File         : train_second_step.py
@Time         : 2021/07/11
@Author       : Ji Jiayu
@Update       : 
@Discription  : A tensorflow implementation of the network in "SCVS:blind image quality assessment based on spatial
correlation and visual saliency"
"""


# from __future__ import unicode_literals
# from __future__ import print_function
# from __future__ import division

import os
import numpy
from scipy import stats
import tensorflow as tf

import scvs, scvs_input

ORDER=numpy.load('order.npy')

def evaluate(type, global_step=None):
    """Evaluation for a number of steps."
    
    Args:
        :param type: str - 'train'/'val'/'test'
        :param global_step: int.
    Returns:
        :return mae: float.
    """
    assert type in ['train', 'val', 'test']

    graph = tf.Graph()
    with graph.as_default() as g:
        keep_prob = tf.placeholder(tf.float32, name='ratio')
        # Input images and labels.
        if type == 'train':
            filenames = [os.path.join(scvs_input.DATA_DIR4, 'image_' + str(6*i+j) + '.tfrecords')
                     for i in ORDER[0:scvs_input.TRAIN_DATA_NUM] for j in range(6)]
            num_data = scvs_input.TRAIN_DATA_NUM * scvs_input.NUM_PER_IMAGE
        elif type == 'val':
            filenames = [os.path.join(scvs_input.DATA_DIR4, 'image_' + str(6*i+j) + '.tfrecords')
                     for i in ORDER[scvs_input.TRAIN_DATA_NUM:scvs_input.TRAIN_DATA_NUM+scvs_input.VAL_DATA_NUM] for j in range(6)]
            num_data = scvs_input.VAL_DATA_NUM * scvs_input.NUM_PER_IMAGE
        elif type == 'test':
            filenames = [os.path.join(scvs_input.DATA_DIR4, 'image_' + str(6*i+j) + '.tfrecords')
                     for i in ORDER[scvs_input.TRAIN_DATA_NUM+scvs_input.VAL_DATA_NUM:
                scvs_input.TRAIN_DATA_NUM + scvs_input.VAL_DATA_NUM + scvs_input.TEST_DATA_NUM] for j in range(6)]
            num_data = scvs_input.TEST_DATA_NUM * scvs_input.NUM_PER_IMAGE

        patches_y, moses, patches_vs = scvs.distorted_inputs_jjy_6(filenames=filenames)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        
        scores = scvs.inference_jjy_4(patches_y, moses, patches_vs, keep_prob)

        total_loss = scvs.loss_func(scores, moses)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            scvs.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()


    with tf.Session(graph=graph) as sess:

        if type == 'val':
            checkpoint_file = os.path.join(scvs_input.LOG_DIR7, 'temp_model.ckpt')
        else:
            checkpoint_file = os.path.join(scvs_input.LOG_DIR7, 'best_model.ckpt')

        saver.restore(sess, checkpoint_file)

        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        score_set = []
        label_set = []
        loss_set = []
        step = 0
        num_iter = int(numpy.ceil(num_data / scvs.BATCH_SIZE))

        while step < num_iter and not coord.should_stop():
            loss_hat, scores_hat, labels_hat = sess.run([total_loss, scores, moses], feed_dict={keep_prob: 1.0})
            score_set.append(scores_hat)
            label_set.append(labels_hat)
            loss_set.append(loss_hat)
            step += 1

        score_set = numpy.reshape(numpy.asarray(score_set), (-1,))
        label_set = numpy.reshape(numpy.asarray(label_set), (-1,))
        loss_set = numpy.reshape(numpy.asarray(loss_set), (-1,))
        print(len(score_set))
        print(len(loss_set))

        # Compute evaluation metric.
        mae = loss_set.mean()
        srocc = stats.spearmanr(score_set, label_set)[0]
        krocc = stats.stats.kendalltau(score_set, label_set)[0]
        plcc = stats.pearsonr(score_set, label_set)[0]
        rmse = numpy.sqrt(((score_set - label_set) ** 2).mean())
        mse = ((score_set - label_set) ** 2).mean()
        print("%s: MAE: %.3f\t SROCC: %.3f\t KROCC: %.3f\t PLCC: %.3f\t RMSE: %.3f\t MSE: %.3f"
              % (type, mae, srocc, krocc, plcc, rmse, mse))
        coord.request_stop()
        coord.join(threads)

    return mae


def train():
    """Train the network for a number of steps.
    
    Args:
        :param order: ndarray - (1,).
    """
    graph = tf.Graph()
    with graph.as_default():
        # Create a variable to count the number of train() calls. For multi-GPU programs, this equals the
        # number of batches processed * num_gpus.
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)
        keep_prob = tf.placeholder(tf.float32, name='ratio')

        # Input images and labels.
        filenames = [os.path.join(scvs_input.DATA_DIR4, 'image_' + str(6*i+j) + '.tfrecords')
                     for i in ORDER[0:scvs_input.TRAIN_DATA_NUM] for j in range(6)]
        patches_y, moses ,patches_vs  = scvs.distorted_inputs_jjy_6(filenames)

        print(patches_y, moses, patches_vs)

        # Build a Graph that computes predictions from the inference model.
        scores = scvs.inference_jjy_4(patches_y,moses,patches_vs, keep_prob)

        # Add to the Graph the Ops for loss calculation.
        total_loss = scvs.loss_func(scores, moses)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op, variable_averages = scvs.train_func(data_num=scvs_input.TRAIN_DATA_NUM*scvs_input.NUM_PER_IMAGE,
                                      total_loss=total_loss, global_step=global_step)

        # Create a saver for writing training checkpoints.
        saver1 = tf.train.Saver()

        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary Tensor based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # The op for initializing the variables.
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

    with tf.Session(graph=graph) as sess:
        # Initialize the variables (the trained variables and the epoch counter).
        sess.run(init_op)

        checkpoint_file = os.path.join('./first_step_logs', 'temp_model.ckpt')

        saver.restore(sess, checkpoint_file)
        sess.run(tf.assign(global_step,0))

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(scvs_input.LOG_DIR7, sess.graph)

        # Start the training loop.
        min_loss = 1000
        best_epoch = 0
        iters_per_epoch = numpy.ceil(scvs_input.TRAIN_DATA_NUM * scvs_input.NUM_PER_IMAGE / scvs.BATCH_SIZE)
        for step in range(scvs.MAX_STEPS):
            _, loss_value = sess.run([train_op, total_loss], feed_dict={keep_prob: 0.5})
            assert not numpy.isnan(loss_value), 'Model diverged with loss = NaN'

            # Write the summaries and print an overview fairly often.
            if step % iters_per_epoch == 0 or (step + 1) == scvs.MAX_STEPS:
                # Print status to stdout.
                print('Epoch %d (Step %d): loss = %.3f' % (step / iters_per_epoch, step, loss_value))
                # Update the events file.
                summary_str = sess.run(summary_op, feed_dict={keep_prob: 0.5})
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # Save a checkpoint and evaluate the model periodically.
            if step % (iters_per_epoch) == 0 or (step + 1) == scvs.MAX_STEPS:
                checkpoint_file = os.path.join(scvs_input.LOG_DIR7, 'temp_model.ckpt')
                saver1.save(sess, checkpoint_file)
                
            if step % (iters_per_epoch*10) == 0 or (step + 1) == scvs.MAX_STEPS:

                val_loss = evaluate('val', step)

                if val_loss < min_loss:
                    min_loss = val_loss
                    best_epoch = step / iters_per_epoch
                    print('best epoch %d with min loss %.3f' % (best_epoch, min_loss))

                    checkpoint_file = os.path.join(scvs_input.LOG_DIR7, 'best_model.ckpt')
                    saver1.save(sess, checkpoint_file)

        coord.request_stop()
        coord.join(threads)

        evaluate(type='test')


if __name__ == '__main__':
    ORDER=numpy.load('order.npy')
    print('data order: %s' % ORDER)

    train()

