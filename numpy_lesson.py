import numpy as np

tc = np.array([25, 28.1, 30.2, 32.5, 35.3, 40.1, 45, 50.3, 55.5, 60.1])
tk = tc * 9 / 5 + 32
a = np.array([1, 2, 3])
print(type(tc)) # <class 'numpy.ndarray'>
print(tc) # [25.  28.1 30.2 32.5 35.3 40.1 45.  50.3 55.5 60.1]
print("Celsius:", tc) # Celsius: [25.  28.1 30.2 32.5 35.3 40.1 45.  50.3 55.5 60.1]
print("Fahrenheit:", tk) # Fahrenheit: [77.   82.58 86.36 90.5  95.54 104.18 113.  122.54 131.9  140.18]

print(tc.ndim) # o'lchamlar soni
print(tc.shape) # (10,) - 10 ta elementli vektor
print(tc.dtype) # float64
print(a.dtype) # int64


# Creating 2D array (matrica)
b = np.array(
    [
    [1, 2, 3, 4],
    [5, 6, 7, 8]
    ]
)
print(b)
print("Array ulchami:", b.ndim) # Array ulchami: 2
print("Array shakli:", b.shape) # Array shakli: (2, 4)
print("Array turi:", b.dtype) # Array turi: int64


tree_dim = np.array([
    [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 90],
        [11, 12, 13, 14, 15]
    ],
    [
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25],
        [26, 27, 28, 29, 30]   
    ]
]
)

print("3D array:", tree_dim.ndim) # 3D array: 3
print("3D array shakli:", tree_dim.shape) # 3D array shakli: (2, 3, 5)
print("3D array turi:", tree_dim.dtype) # 3D array turi: int
print(tree_dim)