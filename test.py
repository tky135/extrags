def num_squares(n: int) -> int:
    cache = {j * j: 1 for j in range(1, int(n ** 0.5) + 1)}

    def numSquares(n: int) -> int:
        
        if n not in cache:
            num_sq = n
            for i in range(1, int(n ** 0.5) + 1):
                num_sq = min(num_sq, numSquares(n - int(i ** 2)) + 1)

            cache[n] = num_sq
        return cache[n]
    result = numSquares(n)
    print(cache)
    return result


print(num_squares(12))  # 3
print(num_squares(13))  # 2