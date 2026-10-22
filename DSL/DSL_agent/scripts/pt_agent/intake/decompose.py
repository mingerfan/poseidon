"""The three specified normalization passes, applied in order."""


def remove_identity(graph):
    aliases = {}
    retained = []
    for node in graph.nodes:
        node.inputs = [aliases.get(name, name) for name in node.inputs]
        if node.op == "identity":
            aliases[node.outputs[0]] = node.inputs[0]
        else:
            retained.append(node)
    graph.outputs = [aliases.get(name, name) for name in graph.outputs]
    graph.nodes = retained
    return graph


def adaptive_to_avgpool(graph):
    for node in graph.nodes:
        if node.op == "adaptive_avgpool":
            h, w = graph.tensors[node.inputs[0]].shape[-2:]
            oh, ow = node.attrs["output_size"]
            if h % oh == 0 and w % ow == 0:
                kernel = (h // oh, w // ow)
                node.op = "avgpool"
                node.attrs = dict(kernel=kernel, stride=kernel, padding=(0, 0))
    return graph


def global_avgpool_flatten(graph):
    removed = set()
    for node in graph.nodes:
        if node.op != "avgpool":
            continue
        shape = graph.tensors[node.inputs[0]].shape
        if len(shape) != 4 or shape[0] != 1:
            continue
        _, channels, height, width = shape
        if (tuple(node.attrs["kernel"]) != (height, width)
                or tuple(node.attrs["stride"]) != (height, width)
                or tuple(node.attrs["padding"]) != (0, 0)
                or node.outputs[0] in graph.outputs):
            continue
        consumers = graph.consumers(node.outputs[0])
        if len(consumers) != 1:
            continue
        consumer = consumers[0]
        matches = ((consumer.op == "reshape" and tuple(consumer.attrs["shape"]) == (1, channels))
                   or (consumer.op == "flatten" and consumer.attrs["start_dim"] == 1
                       and graph.tensors[consumer.outputs[0]].shape == (1, channels)))
        if matches:
            node.op = "mean"
            node.attrs = dict(axes=(2, 3), keepdim=False)
            node.outputs = list(consumer.outputs)
            removed.add(consumer.id)
    graph.nodes = [node for node in graph.nodes if node.id not in removed]
    return graph


def renumber_and_prune(graph):
    used = set(graph.inputs) | set(graph.outputs)
    for index, node in enumerate(graph.nodes):
        node.id = f"n{index}"
        used.update(node.inputs)
        used.update(node.outputs)
    graph.tensors = {name: tensor for name, tensor in graph.tensors.items()
                     if tensor.kind != "act" or name in used}
    return graph


def run(graph):
    return renumber_and_prune(global_avgpool_flatten(adaptive_to_avgpool(remove_identity(graph))))
