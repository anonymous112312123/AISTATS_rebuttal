# AISTATS_rebuttal
This is an anonymous repo for AISTATS rebuttal.

Faster solver for training kernel machines. It accelerates [EigenPro](https://github.com/EigenPro/EigenPro-pytorch) via momentum.

# Test

```python
from axlepro.solvers import *
from axlepro.utils import *

K = LaplacianKernel(bandwidth=1.)
n, d, c, s, q = 3000, 3, 2, 2000, 50
epochs = 60

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X = torch.randn(n, d, device=DEVICE)
y = torch.randn(n, c, device=DEVICE)

X = X * 0.1
X = X.to(DEVICE)
y = y.to(DEVICE)

ahat2, param, time = lm_axlepro_solver(K, X, y, s, q, epochs=epochs, verbose=True)
err2_1 = compute_err(param, K, X, X, y, indices=None, compute_class_err=False)
plt.plot(err2_1, '-b^', label='AxlePro 2, q=50')
print(err2_1)



plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('MSE (train)')
plt.legend()
plt.show()

```
