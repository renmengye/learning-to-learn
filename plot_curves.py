import os
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
  
  sample_rate = 1
  # data_folder = "/home/rjliao/Projects/learning-to-learn/curves/from_1000_eval_100"
  data_folder = "/home/rjliao/Projects/learning-to-learn/curves/from_0_eval_100"

  L2L_loss_file = os.path.join(data_folder, "L2L_loss.csv")
  L2L_lr_file = os.path.join(data_folder, "L2L_learning_rate.csv")
  SGD_loss_file = os.path.join(data_folder, "SGD_loss.csv")

  L2L_loss = read_csv_file(L2L_loss_file)
  L2L_lr = read_csv_file(L2L_lr_file)
  SGD_loss = read_csv_file(SGD_loss_file)

  steps = L2L_loss["Step"][::sample_rate]
  num_steps = len(steps)
  curves = np.zeros([1, num_steps, 2])
  curves[0, :, 0] = L2L_loss["Value"][::sample_rate]
  curves[0, :, 1] = SGD_loss["Value"][::sample_rate]

  plt.figure()
  sns.tsplot(curves, time=steps, condition=["L2L", "SGD"])
  plt.title("Loss vs. Training Step")
  plt.xlabel("Train Step")
  plt.ylabel("Loss")
  plt.savefig(os.path.join(data_folder, "loss.png"))

  curves = np.zeros([1, num_steps, 1])
  curves[0, :, 0] = L2L_lr["Value"][::sample_rate]
  
  plt.figure()
  sns.tsplot(curves, time=steps, condition=["L2L"])
  plt.title("Learning Rate vs. Training Step")
  plt.xlabel("Train Step")
  plt.ylabel("Learning Rate")
  plt.savefig(os.path.join(data_folder, "lr.png"))

if __name__ == "__main__":
  main()