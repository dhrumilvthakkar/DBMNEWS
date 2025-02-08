def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def largest_prime_under(limit):
    for num in range(limit - 1, 1, -1):
        if is_prime(num):
            return num

limit = 1000000
print(f"The largest prime number under {limit} is {largest_prime_under(limit)}")