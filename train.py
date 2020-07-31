import datetime
import contextlib
import tensorflow as tf
import lite
from tensorboard.plugins.hparams import api as hp

# it s recommanded to use absl for tf 2.0
from absl import app
from absl import flags
from absl import logging

import yolact
from data import dataset_coco
from loss import loss_yolact
from utils import learning_rate_schedule

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

tf.random.set_seed(1234)

FLAGS = flags.FLAGS

flags.DEFINE_string('tfrecord_dir', './data/coco',
                    'directory of tfrecord')
flags.DEFINE_string('weights', './weights',
                    'path to store weights')
flags.DEFINE_integer('train_iter', 800000,
                     'iterations')
flags.DEFINE_integer('batch_size', 8,
                     'batch size')
flags.DEFINE_float('lr', 1e-3,
                   'learning rate')
flags.DEFINE_float('momentum', 0.9,
                   'momentum')
# flags.DEFINE_float('weight_decay', 5 * 1e-4,
#                    'weight_decay')
flags.DEFINE_float('print_interval', 10,
                   'number of iteration between printing loss')
flags.DEFINE_float('save_interval', 10000,
                   'number of iteration between saving model(checkpoint)')
flags.DEFINE_float('valid_iter', 5000,
                   'number of iteration between saving validation weights')
class Trainer():
    def __init__(self, model, optimizer, loss_fn, train_metric, val_metric):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_metric = train_metric
        self.val_metric = val_metric

    @tf.function
    def train_step(self, image, labels):
        with tf.GradientTape() as tape:
            pred = self.model(image, training=True)
            output = {
                'seg': pred[0],
                'proto_out': pred[1],
                'pred_cls': pred[2],
                'pred_offset': pred[3],
                'pred_mask_coef': pred[4]
            }

            loc_loss, conf_loss, mask_loss, seg_loss, total_loss = self.loss_fn(output, labels, 13)
        
            grads = tape.gradient(total_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            self.train_metric.update_state(total_loss)
        
        return loc_loss, conf_loss, mask_loss, seg_loss

    @tf.function
    def valid_step(self, image, labels):
        pred = self.model(image, training=False)
        output = {
            'seg': pred[0],
            'proto_out': pred[1],
            'pred_cls': pred[2],
            'pred_offset': pred[3],
            'pred_mask_coef': pred[4]
        }

        loc_loss, conf_loss, mask_loss, seg_loss, total_loss = self.loss_fn(output, labels, 13)
        self.val_metric.update_state(total_loss)
        return loc_loss, conf_loss, mask_loss, seg_loss

    def reset_states(self):
        self.train_metric.reset_states()
        self.val_metric.reset_states()

def main(argv):
    # set up Grappler for graph optimization
    # Ref: https://www.tensorflow.org/guide/graph_optimization
    @contextlib.contextmanager
    def options(options):
        old_opts = tf.config.optimizer.get_experimental_options()
        tf.config.optimizer.set_experimental_options(options)
        try:
            yield
        finally:
            tf.config.optimizer.set_experimental_options(old_opts)

    # -----------------------------------------------------------------
    # Creating dataloaders for training and validation
    logging.info("Creating the dataloader from: %s..." % FLAGS.tfrecord_dir)
    train_dataset = dataset_coco.prepare_dataloader(tfrecord_dir=FLAGS.tfrecord_dir,
                                                    img_size=256,
                                                    batch_size=FLAGS.batch_size,
                                                    subset='train')

    valid_dataset = dataset_coco.prepare_dataloader(tfrecord_dir=FLAGS.tfrecord_dir,
                                                    img_size=256,
                                                    batch_size=1,
                                                    subset='val')
    
    # -----------------------------------------------------------------
    # Creating the instance of the model specified.
    logging.info("Creating the model instance of YOLACT")
    YOLACT = lite.MyYolact(input_size=256,
                          fpn_channels=160,
                          feature_map_size=[32, 16, 8, 4, 2],
                          num_class=13, # 12 classes + 1 background
                          num_mask=32,
                          aspect_ratio=[1, 0.5, 2],
                          scales=[24, 48, 96, 192, 384])

    model = YOLACT.gen()
    

    # add weight decay
    # for layer in model.layers:
    #     if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
    #         layer.add_loss(lambda: tf.keras.regularizers.l2(FLAGS.weight_decay)(layer.kernel))
    #     if hasattr(layer, 'bias_regularizer') and layer.use_bias:
    #         layer.add_loss(lambda: tf.keras.regularizers.l2(FLAGS.weight_decay)(layer.bias))

    # -----------------------------------------------------------------
    # Choose the Optimizor, Loss Function, and Metrics, learning rate schedule
    logging.info("Initiate the Optimizer and Loss function...")
    lr_schedule = learning_rate_schedule.Yolact_LearningRateSchedule(warmup_steps=500, warmup_lr=1e-4, initial_lr=FLAGS.lr)
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete([ 'RMSprop', 'Adam' ]))

    criterion = loss_yolact.YOLACTLoss()
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    valid_loss = tf.keras.metrics.Mean('valid_loss', dtype=tf.float32)
    loc = tf.keras.metrics.Mean('loc_loss', dtype=tf.float32)
    conf = tf.keras.metrics.Mean('conf_loss', dtype=tf.float32)
    mask = tf.keras.metrics.Mean('mask_loss', dtype=tf.float32)
    seg = tf.keras.metrics.Mean('seg_loss', dtype=tf.float32)
    v_loc = tf.keras.metrics.Mean('vloc_loss', dtype=tf.float32)
    v_conf = tf.keras.metrics.Mean('vconf_loss', dtype=tf.float32)
    v_mask = tf.keras.metrics.Mean('vmask_loss', dtype=tf.float32)
    v_seg = tf.keras.metrics.Mean('vseg_loss', dtype=tf.float32)

    with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[HP_OPTIMIZER],
            metrics=[hp.Metric('train_loss', display_name='Train Loss'), hp.Metric('valid_loss', display_name='Valid Loss')],
        )

    # -----------------------------------------------------------------

    # Setup the TensorBoard for better visualization
    # Ref: https://www.tensorflow.org/tensorboard/get_started
    

    # -----------------------------------------------------------------
    

    for trial_idx, optimizer_name in enumerate(HP_OPTIMIZER.domain.values):

        logging.info("Setup the TensorBoard...")
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = './logs/gradient_tape/' + current_time + f'-{optimizer_name}' + '/train'
        test_log_dir = './logs/gradient_tape/' + current_time + f'-{optimizer_name}' + '/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        hparams = {
            HP_OPTIMIZER: optimizer_name
        }

        optimizer_map = {
            'RMSprop': tf.keras.optimizers.RMSprop(learning_rate=lr_schedule, momentum=FLAGS.momentum),
            'Adam': tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        }

        optimizer = optimizer_map[hparams[HP_OPTIMIZER]]

        # Start the Training and Validation Process
        logging.info("Start the training process...")

        # setup checkpoints manager
        checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
        manager = tf.train.CheckpointManager(checkpoint, directory=f'./checkpoints-{optimizer_name}', max_to_keep=5)

        # restore from latest checkpoint and iteration
        status = checkpoint.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            logging.info("Restored from {}".format(manager.latest_checkpoint))
        else:
            logging.info("Initializing from scratch.")

        # Prepare Trainer
        trainer = Trainer(model, optimizer, criterion, train_loss, valid_loss)

        best_val = 1e10
        iterations = checkpoint.step.numpy()

        for image, labels in train_dataset:
            # check iteration and change the learning rate
            if iterations > FLAGS.train_iter:
                break

            checkpoint.step.assign_add(1)
            iterations += 1
            loc_loss, conf_loss, mask_loss, seg_loss = trainer.train_step(image, labels)
            loc.update_state(loc_loss)
            conf.update_state(conf_loss)
            mask.update_state(mask_loss)
            seg.update_state(seg_loss)
            
            with train_summary_writer.as_default():
                tf.summary.scalar('Total loss', trainer.train_metric.result(), step=iterations)
                tf.summary.scalar('Loc loss', loc.result(), step=iterations)
                tf.summary.scalar('Conf loss', conf.result(), step=iterations)
                tf.summary.scalar('Mask loss', mask.result(), step=iterations)
                tf.summary.scalar('Seg loss', seg.result(), step=iterations)

            if iterations and iterations % FLAGS.print_interval == 0:
                logging.info("Iteration {}, LR: {}, Total Loss: {}, B: {},  C: {}, M: {}, S:{} ".format(
                    iterations,
                    optimizer._decayed_lr(var_dtype=tf.float32),
                    trainer.train_metric.result(), loc.result(),
                    conf.result(),
                    mask.result(),
                    seg.result()
                ))

            if iterations and iterations % FLAGS.save_interval == 0:
                # save checkpoint
                save_path = manager.save()
                logging.info("Saved checkpoint for step {}: {}".format(int(checkpoint.step), save_path))
                # validation
                valid_iter = 0
                for valid_image, valid_labels in valid_dataset:
                    if valid_iter > FLAGS.valid_iter:
                        break
                    # calculate validation loss
                    valid_loc_loss, valid_conf_loss, valid_mask_loss, valid_seg_loss = trainer.valid_step(image, labels)
                    v_loc.update_state(valid_loc_loss)
                    v_conf.update_state(valid_conf_loss)
                    v_mask.update_state(valid_mask_loss)
                    v_seg.update_state(valid_seg_loss)
                    valid_iter += 1

                with test_summary_writer.as_default():
                    tf.summary.scalar('V Total loss', trainer.val_metric.result(), step=iterations)
                    tf.summary.scalar('V Loc loss', v_loc.result(), step=iterations)
                    tf.summary.scalar('V Conf loss', v_conf.result(), step=iterations)
                    tf.summary.scalar('V Mask loss', v_mask.result(), step=iterations)
                    tf.summary.scalar('V Seg loss', v_seg.result(), step=iterations)


                with tf.summary.create_file_writer(f'logs/hparam_tuning/trial-{trial_idx}').as_default():
                    hp.hparams(hparams)
                    tf.summary.scalar('train_loss', trainer.train_metric.result(), step=iterations)
                    tf.summary.scalar('valid_loss', trainer.val_metric.result(), step=iterations)

                train_template = 'Iteration {}, Train Loss: {}, Loc Loss: {},  Conf Loss: {}, Mask Loss: {}, Seg Loss: {}'
                valid_template = 'Iteration {}, Valid Loss: {}, V Loc Loss: {},  V Conf Loss: {}, V Mask Loss: {}, Seg Loss: {}'
                logging.info(train_template.format(iterations + 1,
                                            trainer.train_metric.result(),
                                            loc.result(),
                                            conf.result(),
                                            mask.result(),
                                            seg.result()))
                logging.info(valid_template.format(iterations + 1,
                                            trainer.val_metric.result(),
                                            v_loc.result(),
                                            v_conf.result(),
                                            v_mask.result(),
                                            v_seg.result()))
                if trainer.val_metric.result() < best_val:
                    best_val = trainer.val_metric.result()
                    model.save_weights('./weights/weights_' + str(trainer.val_metric.result().numpy()) + '.h5')

        # Reset All States
        trainer.reset_states()
        
        loc.reset_states()
        conf.reset_states()
        mask.reset_states()
        seg.reset_states()
        
        v_loc.reset_states()
        v_conf.reset_states()
        v_mask.reset_states()
        v_seg.reset_states()

            

if __name__ == '__main__':
    app.run(main)
