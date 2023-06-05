import json

data = json.load(open('train-parsed.json', 'r'))


def event_yielder(i):
    for j in i['events']:
        yield j
    yield None


def dia_yielder():
    for i in data:
        print('\nDIALOGUE')
        print('values and counts')
        print('A1: ', i['scenario']['kbs'][0])
        print('A2: ', i['scenario']['kbs'][1])
        yield i
    yield None

d = dia_yielder()
e = event_yielder(next(d))
while input() != 'end':
    item = next(e)
    if item is None:
        print('*'*100)
        print('DIALOGUE DONE')
        print('*'*100)
        e = event_yielder(next(d))
        continue
    print(item['data'])
    print(item['metadata'])
