def combine(first,second):
    return first.broadcast_to(second.shape)+second
