def filter_values(lst):
    if not lst:
        return []

    res = [lst[0]]
    for i in range(1, len(lst)):
        if lst[i] - res[-1] >= 10:
            res.append(lst[i])
    return res

def get_segments(start, end, res):
    segments = []
    for i in res:
        if start < i <= end:
            segments.append([start, i])
            start = i
    if start < end:
        segments.append([start, end])
    return segments

start = 0
end = 1000

res = [126, 128, 138, 145, 174, 807]
new_res = filter_values(res)
new_res = get_segments(start, end, new_res)
print(new_res)  # 输出：[126, 138, 174, 807]