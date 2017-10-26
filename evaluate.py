# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Learning 2 Learn evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import tensorflow as tf

from tensorflow.contrib.learn.python.learn import monitored_session as ms

import os
import meta
import util
import cPickle as pickle

flags = tf.flags
logging = tf.logging

FLAGS = flags.FLAGS
flags.DEFINE_string("optimizer", "L2L", "Optimizer.")
flags.DEFINE_string("path", None, "Path to saved meta-optimizer network.")
flags.DEFINE_integer("num_epochs", 1, "Number of evaluation epochs.")
flags.DEFINE_integer("seed", 1234, "Seed for TensorFlow's RNG.")

flags.DEFINE_string("problem", "simple", "Type of problem.")
flags.DEFINE_integer("num_steps", 100,
                     "Number of optimization steps per epoch.")
flags.DEFINE_float("learning_rate", 0.1, "Learning rate.")
flags.DEFINE_boolean("load_trained_model", False, "Load trained model.")
flags.DEFINE_string("model_path", "exp/model", "Trained model path.")


def main(_):
  # Configuration.
  num_unrolls = FLAGS.num_steps

  if FLAGS.seed:
    tf.set_random_seed(FLAGS.seed)

  # Problem.
  problem, net_config, net_assignments = util.get_config(
      FLAGS.problem, FLAGS.path)

  state = None
  # Optimizer setup.
  if FLAGS.optimizer == "Adam":
    cost_op = problem()
    problem_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    problem_reset = tf.variables_initializer(problem_vars)

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    optimizer_reset = tf.variables_initializer(optimizer.get_slot_names())
    update = optimizer.minimize(cost_op)
    reset = [problem_reset, optimizer_reset]
  elif FLAGS.optimizer == "SGD_MOM":
    cost_op = problem()
    problem_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    problem_reset = tf.variables_initializer(problem_vars)

    optimizer = tf.train.MomentumOptimizer(FLAGS.learning_rate, 0.9)
    optimizer_reset = tf.variables_initializer(optimizer.get_slot_names())
    update = optimizer.minimize(cost_op)
    reset = [problem_reset, optimizer_reset]
  elif FLAGS.optimizer == "L2L":
    if FLAGS.path is None:
      logging.warning("Evaluating untrained L2L optimizer")
    cost_op = problem()
    optimizer = meta.MetaOptimizer(**net_config)
    if FLAGS.load_trained_model:
      # optimizer.load_states([pickle.load(open(os.path.join(FLAGS.model_path, "states.p"), "rb"))])
      # optimizer.load_states([pickle.load(open("./init_state.p", "rb"))])
      meta_loss = optimizer.meta_loss(problem, 1, net_assignments=net_assignments, load_states=False)
      # _, update, reset, cost_op, _ = meta_loss
      _, update, reset, _, _, state = meta_loss
    else:
      meta_loss = optimizer.meta_loss(problem, 1, net_assignments=net_assignments, load_states=False)
      # _, update, reset, cost_op, _ = meta_loss
      _, update, reset, _, _, state = meta_loss 
  else:
    raise ValueError("{} is not a valid optimizer".format(FLAGS.optimizer))

  process_id = os.getpid()
  exp_folder = os.path.join("exp", str(process_id))

  if not os.path.isdir(exp_folder):
    os.mkdir(exp_folder)

  writer = tf.summary.FileWriter(exp_folder)
  summaries = tf.summary.merge_all()

  if FLAGS.problem == "mnist":
    var_name_mlp = [
        "mlp/linear_0/w:0", "mlp/linear_0/b:0", "mlp/linear_1/w:0",
        "mlp/linear_1/b:0"
    ]
  else:
    var_name_mlp = []

  problem_vars = tf.get_collection(tf.GraphKeys.VARIABLES)

  if var_name_mlp:
    saver_vars = [vv for vv in problem_vars if vv.name in var_name_mlp]
  else:
    saver_vars = problem_vars

  saver = tf.train.Saver(saver_vars)

  with ms.MonitoredSession() as sess:
    # a quick hack!
    regular_sess = sess._sess._sess._sess._sess

    # Prevent accidental changes to the graph.
    tf.get_default_graph().finalize()

    if FLAGS.load_trained_model == True:
      print("We are loading trained model here!")
      saver.restore(regular_sess, os.path.join(FLAGS.model_path, "model"))

    # init_state = regular_sess.run(optimizer.init_state)
    # cost_val = regular_sess.run(cost_op)
    # import pdb; pdb.set_trace()

    total_time = 0
    total_cost = 0
    for step in xrange(FLAGS.num_epochs):
      # Training.
      # time, cost = util.run_epoch(sess, cost_op, [update], reset,
      #                             num_unrolls)

      time, cost, final_states = util.run_epoch_eval(
          sess,
          cost_op, [update],
          reset,
          num_unrolls,
          state_ops=state,
          summary_op=summaries,
          summary_writer=writer,
          run_reset=False)
      writer.flush()

      total_time += time
      total_cost += cost

    saver.save(regular_sess, os.path.join(exp_folder, "model"))
    pickle.dump(final_states, open(os.path.join(exp_folder, "states.p"), "wb"))

    # Results.
    util.print_stats("Epoch {}".format(FLAGS.num_epochs), total_cost,
                     total_time, FLAGS.num_epochs)

    # we have to run in the end to skip the error
    regular_sess.run(reset)

if __name__ == "__main__":
  tf.app.run()
