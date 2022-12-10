import numpy as np
import time

n = 1000

np.random.seed(0)
y_ = np.random.random(n)
test_y = np.random.random(n)

t_start = time.time()
s1 = 0
n = len(y_)
for i in range(n - 1):
  for j in range(i + 1, n):
    h = np.sign(y_[i] - y_[j])
    test_h = np.sign(test_y[i] - test_y[j])
    if h == test_h:
      s1 += 1
s1 = 2 * s1 / (n * (n - 1))
print(f"{time.time() - t_start:.4} sec")

t_start = time.time()
idx = np.triu_indices(n, 1)
y_diffs = y_.reshape(-1, 1) - y_
test_y_diffs = test_y.reshape(-1, 1) - test_y
y_comparison = np.sign(y_diffs[idx])
test_y_comparison = np.sign(test_y_diffs[idx])
s2 = np.mean(y_comparison == test_y_comparison)
print(f"{time.time() - t_start:.4} sec")

assert s1 == s2
