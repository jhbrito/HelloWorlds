# this is a comment

print("Hello World")

a = 1
b = 2
c = a + b
print("c=", c)

c = 4

if c == 3:
    print("entrei no if")
elif c == 4:
    print("entrei no elif")
else:
    print("entrei no else")

y = 1
for x in range(10):
    print("x =", x)
    y = y + x
print("y =", y)

x = 0
y = 0
while x < 10:
    print("x =", x)
    y = y + x
    x = x + 1
print("y =", y)

a = 1
b = 3.1415
c = a + b

a = 0.1
b = 0.2
c = a + b

m = "olá"
print(m)
print(m[0:1])
o = "adeus"
p = m + " " + o
print("p:", p)


def f1(x1, x2):
    yf1 = x1 * x2
    return yf1


q = f1(10, 3)
print("q:", q)

a = (1, 3, 5, 7, 9)
print("a")
print(a)
print(a[3])
print(a[2:4])
print(a[2:])
# a[1]=10

b = [2, 4, 6, 8, 10]
print("b")
print(b)
print(b[3])
print(b[2:4])
print(b[2:])
b[1] = 10

c = list(a)
print("c")
print(c)
print(c[3])
print(c[2:4])
print(c[2:])
c[1] = 12

# sets - unordered collection with no duplicate elements
d = {1, 2, 3, 4, 3, 5}
print(d)
for i in range(10):
    if i in d:
        print(str(i), " is in d")

# dictionaries
words = dict()
words["BCVT"] = "Brito Computer Vision Team"
words["LEEC"] = "Licenciatura em Engenharia Electrotécnica e de Computadores"
words["EST"] = "Escola Superior de Tecnologia"
words["IPCA"] = "Instituto Politécnico do Cávado e do Ave"

print("BCVT - ", words["BCVT"])

print("Goodbye World")
