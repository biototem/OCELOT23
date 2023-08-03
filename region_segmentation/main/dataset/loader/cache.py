from basic import config

MAX_LENGTH = config['ddataset.cache.length'] or 20
MAX_MEMORY = config['ddataset.cache.memory'] or 4      # GB


i = 0
que = [None] * MAX_LENGTH
data = {}


def register(key: str, generator: callable):
    data[key] = {
        'gen': generator,
        'created': False,
        'lock': False,
        'index': None,
        'data': None,
    }


# This is Dynamic query & load method -> using method registered before
def query(key: str):
    global i
    d = data[key]
    while d['lock']:
        print('what?')
    if not d['created']:
        d['lock'] = True
        d['data'] = d['gen']()
        d['index'] = i
        pop_k = que[i]
        que[i] = key
        if pop_k is not None:
            data[pop_k]['index'] = None
            data[pop_k]['data'] = None
            data[pop_k]['created'] = False
        i = (i + 1) % MAX_LENGTH
        d['lock'] = False
        d['created'] = True
    return d['data']
