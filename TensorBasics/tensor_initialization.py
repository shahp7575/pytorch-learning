import torch

# =================================================================== #
# =================== INITIALIZING TENSOR BASICS ==================== #
# =================================================================== #

# 2 rows 3 cols
my_tensor = torch.tensor([[1,2,3], [4,5,6]])

# optionally set dtype as well as device the tensor should be on (cuda/cpu)
my_tensor = torch.tensor([[1,2,3], [4,5,6]], dtype=torch.float32,
                         device='cuda')

# good practice for setting up device
device = "cuda" if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1,2,3], [4,5,6]], dtype=torch.float32,
                         device=device)

# print tensor
print(my_tensor)

# print tensor dtype
print(my_tensor.dtype)

# print tensor device
print(my_tensor.device)

# print tensor shape
print(my_tensor.shape)

# ============================================================================ #
# =================== OTHER COMMON INITIALIZATION METHODS ==================== #
# ============================================================================ #

##### if in case we don't have exact values for the tensor
# empty 3x3 tensor
X = torch.empty(size = (3,3))

# 3x3 zeros
X = torch.zeros(size = (3,3))

# 3x3 random with values from uniform distribution
X = torch.rand(size = (3,3))

# 3x3 ones
X = torch.ones(size = (3,3))

# 3x3 identity matrix
X = torch.eye(3,3)

# arange -> similar to range python. start with 0 up till (end-1) with an increment of 1
X = torch.arange(start=0, end=5, step=1)

# linspace -> start at 0.1, end at 1; with 10 values between those
X = torch.linspace(start=0.1, end=1, steps=10)

# empty with normal distribution
X = torch.empty(size=(1,5)).normal_(mean=0, std=1)

# empty with uniform distribution; similar to torch.rand but here you can specify
# the lower and upper of the uniform distribution
X = torch.empty(size=(1,5)).uniform_(0, 1)

# diagonal matrix with 1's in the diagonal
X = torch.diag(torch.ones(3))


# ================================================================================= #
# ============= HOW TO INITIALIZE AND CONVERT TENSORS TO OTHER TYPES ============== #
# ================================================================================= #

tensor = torch.arange(4)
# convert to boolean
print(tensor.bool())
# to int16
print(tensor.short())
# to int64
print(tensor.long())
# to float16
print(tensor.half())
# to float32
print(tensor.float())
# to double
print(tensor.double())

# ================================================================================= #
# =================== ARRAY TO TENSOR CONVERSION AND VICE-VERSA =================== #
# ================================================================================= #
import numpy as np
np_array = np.zeros((5, 5))
# numpy array to tensor
tensor = torch.from_numpy(np_array)
# tensor to numpy back
np_array_back = tensor.numpy()

# ================================================================================= #
# ======================= TENSOR MATH & COMPARISON OPERATORS ====================== #
# ================================================================================= #

# init two tensors
x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])

# addition
z1 = torch.empty(3)
torch.add(x, y, out=z1)
print("Z1 -> ", z1)

z2 = torch.add(x, y)
print("Z2 -> ", z2)

z3 = x + y
print("Z3 -> ", z3)

# subtraction
z = x - y
print("Z -> ", z)

# element wise division
z = torch.true_divide(x, y)
print("Z -> ", z)

# in-place operation
t = torch.zeros(3)
# all operations with _ are in-place. so it doesn't create an extra tensor but mutates in place
t.add_(x)
# another way
t += x

# exponentiation
z = x.pow(2)
## another-way
z = x**2

# Simple Comparison
z = x > 0

# Matrix Multiplication
x1 = torch.rand((2,5))
x2 = torch.rand((5,3))

x3 = torch.mm(x1,x2) # 2x3

## another way
x3 = x1.mm(x2)

# Matrix Exponentiation. (matrix mutiplication x times)
matrix_exp = torch.rand(5, 5)
matrix_exp.matrix_power(3)

# Element-wise multiplication
z = x * y

# dot product
z = torch.dot(x, y)
print(z)

# batch matrix multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))

out_bmm = torch.bmm(tensor1, tensor2)

# Examples of broadcasting
x1 = torch.rand((5,5))
x2 = torch.rand((1,5))
z = x1 - x2

# Other useful tensor operations
# You can specify which dimension it should sum over
sum_x = torch.sum(x, dim=0)
# value and index of max/min
values, index = torch.max(x, dim=0) # also x.max(dim=0)
values, index = torch.min(x, dim=0)

# absolute value
abs_x = torch.abs(x)

# argmax/argmin; same as max except it only returns the index of max/min value
z = torch.argmax(x, dim=0)
z = torch.argmin(x, dim=0)

# mean; to compute a mean, pytorch requires it to be a float
mean_x = torch.mean(x.float(), dim=0)

# Element-wise Compare
## check which values are equal (bool)
z = torch.eq(x, y)

# Sort
sorted_y, indices = torch.sort(y, dim=0, descending=False)

# Clamp; All elements that are less than 0 (min), will be set to 0.
# Elements greater than 10 (max), will be set to 10.
z = torch.clamp(x, min=0, max=10)

# check if any values are true
x = torch.tensor([1,0,1,1,1], dtype=torch.bool)
z = torch.any(x)

# check if all values are true
z = torch.all(x)


# ================================================================================= #
# ================================ TENSOR INDEXING ================================ #
# ================================================================================= #

# Let's say we have a batch size of 10
# and 25 features of every example in our batch
batch_size = 10
features = 25
X = torch.rand((batch_size, features))

### print 1st example
print(X[0])

### print 1st feature of all examples
print(X[:, 0])

### get the 3rd example from the batch and 1st ten features
print(X[2, :10])

# Fancy Indexing
X = torch.arange(10)
indices = [2,5,8]
print(X[indices])

X = torch.rand((3,5))
rows = torch.tensor([1,0])
cols = torch.tensor([4,0])
print(X[rows, cols])

# More Advanced Indexing
X = torch.arange(10)
## pick all the elements that are less than 2 and greater than 8
print(X[(X < 2) | (X > 8)])

# Useful Operations
## if the value of x is greater than 5 then return x, else x*2
print(torch.where(X > 5, X, X*2))

## unique
print(torch.tensor([0,0,1,1,2,2,3,4]).unique())

# ================================================================================== #
# ================================ TENSOR RESHAPING ================================ #
# ================================================================================== #

X = torch.arange(9)
# convert X to a 3x3 matrix
x_3x3 = X.view(3, 3)
# Another-way using .reshape
x_3x3 = X.reshape(3, 3)


x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))
# concatenate both across dimension
print(torch.cat((x1, x2), dim=0).shape)

# unroll x1; flatten the entire thing 2x5 to 10
z = x1.view(-1)

batch = 64
X = torch.rand((batch, 2, 5))
# keep the batch dimension but flatten the rest (2,5)
z = X.view(batch, -1)

# switch axis; (batch, 2, 5) -> (batch, 5, 2)
z = X.permute(0, 2, 1)

X = torch.arange(10) # [10]
# make to a 1x10 vector
print(X.unsqueeze(0))
# make a 10x1 vector
print(X.unsqueeze(1))


