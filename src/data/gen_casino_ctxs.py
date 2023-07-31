tmp = [  
    [3, 4, 5],
    [3, 5, 4],
    [4, 3, 5],
    [4, 5, 3],
    [5, 3, 4],
    [5, 4, 3],
]

ctx_getter = lambda x: f'3 {x[0]} 3 {x[1]} 3 {x[2]}\n'

out_f = 'selfplay.txt'

with open(out_f, 'w') as f:
    for c in tmp:
        for d in tmp:
            f.write(ctx_getter(c))
            f.write(ctx_getter(d))
            