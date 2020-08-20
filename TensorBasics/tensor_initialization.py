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
print(X)

