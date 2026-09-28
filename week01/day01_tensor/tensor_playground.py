import torch
print("Pytorch Version:", torch.__version__)

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print("Device:", device)


x = torch.randn(2, 3, 4)
print(x)
print("x.shape:", x.shape)
print("ndim:", x.ndim)
print("dtype:", x.dtype)

y = x.reshape(2, 12)
print("y.shape:", y.shape)

z = x.transpose(1, 2)
print("z.shape:", z.shape)

p = x.permute(0 ,2, 1)
print("p.shape:", p.shape)

u = x.unsqueeze(1)
print(u)
print("u.shape:", u.shape)


a = torch.randn(5, 10, 20) # a.shape = [5, 10, 20]
b = a.transpose(1, 2) # b.shape = [5, 20, 10]
c = a.permute(2, 0, 1) # c.shape = [20, 5, 10]
d = a.unsqueeze(0) # d.shape = [1, 5, 10, 20]
print(a.shape)
print(b.shape)
print(c.shape)
print(d.shape)


a = torch.randn(2, 3, 4)
b = torch.randn(4)
c = a + b
print("a:", a.shape)
print("b:", b.shape)
print("c:", c.shape)

a = torch.randn(2, 3, 4)
b = torch.randn(1, 3, 1)
c = a + b
print("a:", a.shape)
print("b:", b.shape)
print("c:", c.shape)

# Matrix Multiplication
B = 2
T = 3
D = 4
H = 8

X = torch.randn(B, T, D)
W = torch.randn(D, H)
Y = X @ W
print("X:", X.shape)
print("W:", W.shape)
print("Y:", Y.shape)


# Attention
B = 2
T = 5
D = 8

Q = torch.randn(B, T, D)
K = torch.randn(B, T, D)

scores = Q @ K.transpose(-2, -1)

print("Q:", Q.shape)
print("K:", K.shape)
print("K transpose:", K.transpose(-2, -1).shape)
print("scores:", scores.shape)