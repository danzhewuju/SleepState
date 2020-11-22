import math


def mcm(num):
    minimum = 1
    for i in num:
        minimum = int(i) * int(minimum) / math.gcd(int(i), int(minimum))
    return int(minimum)


print(mcm([1, 3, 5, 8]))
