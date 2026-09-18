"""Bounded public string operations; never dispatch user-defined methods."""

METHODS = {
    'strip': str.strip, 'lstrip': str.lstrip, 'rstrip': str.rstrip,
    'split': str.split, 'rsplit': str.rsplit,
    'partition': str.partition, 'rpartition': str.rpartition,
    'replace': str.replace, 'join': str.join,
}


def invoke(name, receiver, arguments, keywords):
    if name not in METHODS or type(receiver) is not str or len(receiver) > 128:
        raise ValueError('String method requires a bounded public string receiver')
    if type(arguments) is not list or type(keywords) is not dict:
        raise ValueError('Invalid string argument container')
    if len(arguments)+len(keywords) > 128 or not all(type(k) is str for k in keywords):
        raise ValueError('String argument resource/type limit')
    # Freeze pinned Python 3.10 signatures even if the host checker uses a
    # newer Python (replace(count=...) was added in Python 3.13).
    if keywords and (name not in ('split','rsplit') or not set(keywords) <= {'sep','maxsplit'}):
        raise ValueError('Unsupported keyword for pinned string signature')
    for value in [*arguments,*keywords.values()]:
        if name == 'join' and type(value) is list:
            if len(value) > 128 or not all(type(v) is str and len(v) <= 128 for v in value):
                raise ValueError('join requires bounded public strings')
        elif type(value) not in (type(None),str,int,bool):
            raise ValueError('String arguments must be public strings/integers/None')
        elif type(value) is str and len(value) > 128:
            raise ValueError('String argument length limit')
        elif type(value) is int and abs(value) > 1048576:
            raise ValueError('String integer argument limit')
    if name == 'join':
        if len(arguments) != 1 or keywords or type(arguments[0]) is not list:
            raise ValueError('join requires one positional public iterable')
        items = arguments[0]
        size = sum(len(s) for s in items)+len(receiver)*max(0,len(items)-1)
        if size > 128:
            raise ValueError('String result length limit')
    try:
        result = METHODS[name](receiver,*arguments,**keywords)
    except (TypeError,ValueError,OverflowError) as error:
        raise ValueError('Invalid public string method arguments') from error
    values = result if type(result) in (list,tuple) else [result]
    if (type(result) in (list,tuple) and len(result) > 128 or
        not all(type(v) is str and len(v) <= 128 for v in values)):
        raise ValueError('String result resource limit')
    return result
