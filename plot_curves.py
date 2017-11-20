import os
import glob
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def read_csv_file(file_name):
  with open(file_name, "r") as ff:
    count = 0

    for line in ff:
      line_str = line.rstrip().split(",")

      if count == 0:
        results = {}
        key_list = line_str
        for key in key_list:
          results[key] = []

      else:
        for ii, xx in enumerate(line_str):
          results[key_list[ii]] += [float(xx)]

      count += 1

  return results


def main():

  sample_rate = 10
  data_folder = "/home/rjliao/Projects/learning-to-learn/curves/relu_network_100_hidden"

  # plot loss
  loss_file_list = glob.glob(os.path.join(data_folder, "*_loss.csv"))

  loss_val = []
  conditions = []
  for ff in loss_file_list:
    conditions += [ff.split("/")[-1][:-9].replace("_", " ")]
    loss_val += [np.array(read_csv_file(ff)["Value"])]

  curves = np.concatenate(
      [np.expand_dims(xx, axis=1) for xx in loss_val], axis=1)
  curves = np.expand_dims(curves, axis=0)
  steps = range(0, 10000, 10)
  curves = np.log(curves[:, ::sample_rate, :])
  steps = steps[::sample_rate]

  plt.figure()
  sns.tsplot(curves, time=steps, condition=conditions)
  plt.title("Loss vs. Training Step")
  plt.xlabel("Train Step")
  plt.ylabel("Loss")
  plt.savefig(os.path.join(data_folder, "log_loss.png"))

  lr_file_list = glob.glob(os.path.join(data_folder, "*_lr.csv"))

  lr_val = []
  conditions = []
  for ff in lr_file_list:
    conditions += [ff.split("/")[-1][:-7].replace("_", " ")]
    lr_val += [np.array(read_csv_file(ff)["Value"])]

  # import pdb; pdb.set_trace()
  curves = np.concatenate([np.expand_dims(xx, axis=1) for xx in lr_val], axis=1)
  curves = np.expand_dims(curves, axis=0)
  steps = range(0, 10000, 10)
  curves = curves[:, ::sample_rate, :]
  steps = steps[::sample_rate]

  plt.figure()
  sns.tsplot(curves, time=steps, condition=conditions)
  plt.title("Learning Rate vs. Training Step")
  plt.xlabel("Train Step")
  plt.ylabel("Learning Rate")
  plt.savefig(os.path.join(data_folder, "lr.png"))


if __name__ == "__main__":
  main()
