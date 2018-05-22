"""
Train PDDM models
"""

from datetime import datetime
import os
import time
import sys
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import itertools
import random
import pdb
from six import iteritems
from sklearn.metrics import average_precision_score
import glob

sys.path.append('../')
from configs.train_config import TrainConfig
from data_io import session_generator, load_data_and_label, prepare_dataset
import networks
import utils


"""
Reference:
    FaceNet implementation:
    https://github.com/davidsandberg/facenet
"""
def main():

    cfg = TrainConfig().parse()
    print (cfg.name)
    result_dir = os.path.join(cfg.result_root, 
            cfg.name+'_'+datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S'))
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    utils.write_configure_to_file(cfg, result_dir)
    np.random.seed(seed=cfg.seed)

    # prepare dataset
    train_session = cfg.train_session
    train_set = prepare_dataset(cfg.feature_root, train_session, cfg.feat, cfg.label_root)
    train_set = train_set[:cfg.label_num]
    batch_per_epoch = len(train_set)//cfg.sess_per_batch

    val_session = cfg.val_session
    val_set = prepare_dataset(cfg.feature_root, val_session, cfg.feat, cfg.label_root)


    # construct the graph
    with tf.Graph().as_default():
        tf.set_random_seed(cfg.seed)
        global_step = tf.Variable(0, trainable=False)
        lr_ph = tf.placeholder(tf.float32, name='learning_rate')

        # load backbone model
        if cfg.network == "tsn":
            model_emb = networks.TSN(n_seg=cfg.num_seg, emb_dim=cfg.emb_dim)
        elif cfg.network == "rtsn":
            model_emb = networks.RTSN(n_seg=cfg.num_seg, emb_dim=cfg.emb_dim, n_input=cfg.n_input)
        elif cfg.network == "convtsn":
            model_emb = networks.ConvTSN(n_seg=cfg.num_seg, emb_dim=cfg.emb_dim)
        elif cfg.network == "convrtsn":
            model_emb = networks.ConvRTSN(n_seg=cfg.num_seg, emb_dim=cfg.emb_dim, n_h=cfg.n_h, n_w=cfg.n_w, n_C=cfg.n_C, n_input=cfg.n_input)
        elif cfg.network == "convbirtsn":
            model_emb = networks.ConvBiRTSN(n_seg=cfg.num_seg, emb_dim=cfg.emb_dim)
        else:
            raise NotImplementedError
        model_ver = networks.PDDM(n_input=cfg.emb_dim)

        # get the embedding
        if cfg.feat == "sensors" or cfg.feat == "segment":
            input_ph = tf.placeholder(tf.float32, shape=[None, cfg.num_seg, None])
        elif cfg.feat == "resnet" or cfg.feat == "segment_down":
            input_ph = tf.placeholder(tf.float32, shape=[None, cfg.num_seg, None, None, None])
        dropout_ph = tf.placeholder(tf.float32, shape=[])
        model_emb.forward(input_ph, dropout_ph)
        if cfg.normalized:
            embedding = tf.nn.l2_normalize(model_emb.hidden, axis=-1, epsilon=1e-10)
        else:
            embedding = model_emb.hidden

        # variable for visualizing the embeddings
        emb_var = tf.Variable([0.0], name='embeddings')
        set_emb = tf.assign(emb_var, embedding, validate_shape=False)

        # calculated for monitoring all-pair embedding distance
        diffs = utils.all_diffs_tf(embedding, embedding)
        all_dist = utils.cdist_tf(diffs)
        tf.summary.histogram('embedding_dists', all_dist)

        # split embedding into anchor, positive and negative and calculate triplet loss
        anchor, positive, negative = tf.unstack(tf.reshape(embedding, [-1,3,cfg.emb_dim]), 3, 1)
        metric_loss = networks.triplet_loss(anchor, positive, negative, cfg.alpha)

        model_ver.forward(tf.stack((anchor, positive), axis=1))
        pddm_ap = model_ver.prob[:, 0]
        model_ver.forward(tf.stack((anchor, negative), axis=1))
        pddm_an = model_ver.prob[:, 0]
        pddm_loss = tf.reduce_mean(tf.maximum(tf.add(tf.subtract(pddm_ap, pddm_an), 0.6), 0.0), 0)

        regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss = pddm_loss + 0.5 * metric_loss + regularization_loss * cfg.lambda_l2

        tf.summary.scalar('learning_rate', lr_ph)
        train_op = utils.optimize(total_loss, global_step, cfg.optimizer,
                lr_ph, tf.global_variables())

        saver = tf.train.Saver(max_to_keep=10)

        summary_op = tf.summary.merge_all()

        # session iterator for session sampling
        feat_paths_ph = tf.placeholder(tf.string, shape=[None, cfg.sess_per_batch])
        label_paths_ph = tf.placeholder(tf.string, shape=[None, cfg.sess_per_batch])
        train_data = session_generator(feat_paths_ph, label_paths_ph, sess_per_batch=cfg.sess_per_batch, num_threads=2, shuffled=False, preprocess_func=model_emb.prepare_input)
        train_sess_iterator = train_data.make_initializable_iterator()
        next_train = train_sess_iterator.get_next()

        # prepare validation data
        val_feats = []
        val_labels = []
        for session in val_set:
            eve_batch, lab_batch, _ = load_data_and_label(session[0], session[1], model_emb.prepare_input_test)    # use prepare_input_test for testing time
            val_feats.append(eve_batch)
            val_labels.append(lab_batch)
        val_feats = np.concatenate(val_feats, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        print ("Shape of val_feats: ", val_feats.shape)

        # generate metadata.tsv for visualize embedding
        with open(os.path.join(result_dir, 'metadata_val.tsv'), 'w') as fout:
            for v in val_labels:
                fout.write('%d\n' % int(v))


        # Start running the graph
        if cfg.gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        summary_writer = tf.summary.FileWriter(result_dir, sess.graph)

        with sess.as_default():

            sess.run(tf.global_variables_initializer())

            # load pretrain model, if needed
            if cfg.model_path:
                print ("Restoring pretrained model: %s" % cfg.model_path)
                saver.restore(sess, cfg.model_path)

            ################## Training loop ##################
            epoch = -1
            while epoch < cfg.max_epochs-1:
                step = sess.run(global_step, feed_dict=None)
                epoch = step // batch_per_epoch

                # learning rate schedule, reference: "In defense of Triplet Loss"
                if epoch < cfg.static_epochs:
                    learning_rate = cfg.learning_rate
                else:
                    learning_rate = cfg.learning_rate * \
                            0.001**((epoch-cfg.static_epochs)/(cfg.max_epochs-cfg.static_epochs))

                # prepare data for this epoch
                random.shuffle(train_set)

                feat_paths = [path[0] for path in train_set]
                label_paths = [path[1] for path in train_set]
                # reshape a list to list of list
                # interesting hacky code from: https://stackoverflow.com/questions/10124751/convert-a-flat-list-to-list-of-list-in-python
                feat_paths = list(zip(*[iter(feat_paths)]*cfg.sess_per_batch))
                label_paths = list(zip(*[iter(label_paths)]*cfg.sess_per_batch))

                sess.run(train_sess_iterator.initializer, feed_dict={feat_paths_ph: feat_paths,
                  label_paths_ph: label_paths})

                # for each epoch
                batch_count = 1
                while True:
                    try:
                        # Hierarchical sampling (same as fast rcnn)
                        start_time_select = time.time()

                        # First, sample sessions for a batch
                        eve, se, lab = sess.run(next_train)

                        select_time1 = time.time() - start_time_select

                        # Get the similarity of all events
                        sim_prob = np.zeros((eve.shape[0], eve.shape[0]), dtype='float32')*np.nan
                        comb = list(itertools.combinations(range(eve.shape[0]), 2))
                        for start, end in zip(range(0, len(comb), cfg.batch_size),
                                            range(cfg.batch_size, len(comb)+cfg.batch_size, cfg.batch_size)):
                            end = min(end, len(comb))
                            comb_idx = []
                            for c in comb[start:end]:
                                comb_idx.extend([c[0], c[1], c[1]])
                            emb = sess.run(pddm_ap, feed_dict={input_ph: eve[comb_idx], dropout_ph: 1.0})
                            for i in range(emb.shape[0]):
                                sim_prob[comb[start+i][0], comb[start+i][1]] = emb[i]
                                sim_prob[comb[start+i][1], comb[start+i][0]] = emb[i]

                        # Second, sample triplets within sampled sessions
                        triplet_selected, active_count = utils.select_triplets_facenet(lab,sim_prob,cfg.triplet_per_batch,cfg.alpha)

                        select_time2 = time.time()-start_time_select-select_time1


                        start_time_train = time.time()
                        triplet_input_idx = [idx for triplet in triplet_selected for idx in triplet]
                        triplet_input = eve[triplet_input_idx]
                        # perform training on the selected triplets
                        err, _, step, summ = sess.run([total_loss, train_op, global_step, summary_op],
                                feed_dict = {input_ph: triplet_input,
                                            dropout_ph: cfg.keep_prob,
                                            lr_ph: learning_rate})

                        train_time = time.time() - start_time_train
                        print ("%s\tEpoch: [%d][%d/%d]\tEvent num: %d\tTriplet num: %d\tSelect_time1: %.3f\tSelect_time2: %.3f\tTrain_time: %.3f\tLoss %.4f" % \
                                (cfg.name, epoch+1, batch_count, batch_per_epoch, eve.shape[0], triplet_input.shape[0]//3, select_time1, select_time2, train_time, err))

                        summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=err),
                            tf.Summary.Value(tag="active_count", simple_value=active_count),
                            tf.Summary.Value(tag="triplet_num", simple_value=triplet_input.shape[0]//3)])
                        summary_writer.add_summary(summary, step)
                        summary_writer.add_summary(summ, step)

                        batch_count += 1
                    
                    except tf.errors.OutOfRangeError:
                        print ("Epoch %d done!" % (epoch+1))
                        break

                # validation on val_set
                print ("Evaluating on validation set...")
                val_embeddings, _ = sess.run([embedding, set_emb], feed_dict={input_ph: val_feats, dropout_ph: 1.0})
                mAP, mPrec = utils.evaluate_simple(val_embeddings, val_labels)


                val_sim_prob = np.zeros((val_feats.shape[0], val_feats.shape[0]), dtype='float32')*np.nan
                val_comb = list(itertools.combinations(range(val_feats.shape[0]), 2))
                for start, end in zip(range(0, len(val_comb), cfg.batch_size),
                                    range(cfg.batch_size, len(val_comb)+cfg.batch_size, cfg.batch_size)):
                    end = min(end, len(val_comb))
                    comb_idx = []
                    for c in val_comb[start:end]:
                        comb_idx.extend([c[0], c[1], c[1]])
                    emb = sess.run(pddm_ap, feed_dict={input_ph: val_feats[comb_idx], dropout_ph: 1.0})
                    for i in range(emb.shape[0]):
                        val_sim_prob[val_comb[start+i][0], val_comb[start+i][1]] = emb[i]
                        val_sim_prob[val_comb[start+i][1], val_comb[start+i][0]] = emb[i]

                mAP_PDDM = 0.0
                count = 0
                for i in range(val_labels.shape[0]):
                    if val_labels[i] > 0:
                        temp_labels = np.delete(val_labels, i, 0)
                        temp = np.delete(val_sim_prob, i, 1)
                        mAP_PDDM += average_precision_score(np.squeeze(temp_labels==val_labels[i,0]),
                                        np.squeeze(1 - temp[i]))
                        count += 1
                mAP_PDDM /= count

                summary = tf.Summary(value=[tf.Summary.Value(tag="Validation mAP", simple_value=mAP),
                                            tf.Summary.Value(tag="Validation mAP_PDDM", simple_value=mAP_PDDM),
                                            tf.Summary.Value(tag="Validation mPrec@0.5", simple_value=mPrec)])
                summary_writer.add_summary(summary, step)
                print ("Epoch: [%d]\tmAP: %.4f\tmPrec: %.4f\tmAP_PDDM: %.4f" % (epoch+1,mAP,mPrec,mAP_PDDM))

                # config for embedding visualization
                config = projector.ProjectorConfig()
                visual_embedding = config.embeddings.add()
                visual_embedding.tensor_name = emb_var.name
                visual_embedding.metadata_path = os.path.join(result_dir, 'metadata_val.tsv')
                projector.visualize_embeddings(summary_writer, config)

                # save model
                saver.save(sess, os.path.join(result_dir, cfg.name+'.ckpt'), global_step=step)

if __name__ == "__main__":
    main()
