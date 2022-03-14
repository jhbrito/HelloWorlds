import numpy as np

# Simple Vector
a = np.array([1, 2, 3])
print("a", a)
print("type", type(a))
print("a[0]", a[0])

# Simple Matrix and indexing
a=np.array([[1, 2, 3], [4, 5, 6]])
print("a", a)
print("a[0,1]", a[0,1])
print("a[0][1]", a[0][1])

# Matrix properties
a = np.arange(15).reshape(3, 5)
print("a.shape", a.shape)
print("a.ndim", a.ndim)
print("a.dtype", a.dtype)
print("a.dtype.name", a.dtype.name)
print("a.itemsize", a.itemsize)
print("a.size", a.size)
print("a", a)

# Special matrices
a=np.empty((3, 4))
print("a", a)
b=np.zeros((3, 4))
print("b", b)
c=np.ones((3, 4))
print("c", c)
d=np.eye(3)
print("d", d)
e=np.random.randn(12).reshape((3,4))
print("e", e)

# Simple Statistics
print("Max", np.max(a))
print("Min", np.min(b))
print("Mean", np.mean(c))
print("Std Dev", np.std(d))

# Broadcasting
a = np.array([1.0, 2.0, 3.0, 4.0])
b = np.array([2.0, 2.0, 2.0, 2.0])
print("a:{}".format(a))
print("b:{}".format(b))
print("a+b:{}".format(a+b))
print("a*b:{}".format(a*b))
print("a**b:{}".format(a**b))

b = np.array([2.0])
print("a:{}".format(a))
print("b:{}".format(b))
print("a+b:{}".format(a+b))
print("a*b:{}".format(a*b))
print("a**b:{}".format(a**b))

a = np.array([[1.0, 2.0], [3.0, 4.0]])
b = np.array([1.0, 2.0])
print("a:{}".format(a))
print("b:{}".format(b))
print("a+b:{}".format(a+b))
print("b+a:{}".format(b+a))
print("a*b:{}".format(a*b))
print("b*a:{}".format(b*a))
print("a**b:{}".format(a**b))

a = np.array([[1.0, 2.0], [3.0, 4.0]])
b = np.array([[1.0], [2.0]])
print("a:{}".format(a))
print("b:{}".format(b))
print("a+b:{}".format(a+b))
print("b+a:{}".format(b+a))
print("a*b:{}".format(a*b))
print("b*a:{}".format(b*a))
print("a**b:{}".format(a**b))

a = np.random.randn(4).reshape((2,2))
b = np.random.randn(4).reshape((2,2))

c = a @ b
print("c:{}".format(c))

d = 2 * a
print("d:{}".format(d))

