"""Annotate the graph patterns supported by the trusted lowering."""

from .graph import Graph


def annotate(graph: Graph) -> list[dict]:
    """Return pattern groups numbered by their first node's graph order."""
    groups = []

    def only_consumer(node):
        consumers = graph.consumers(node.outputs[0])
        return consumers[0] if len(consumers) == 1 else None

    for node in graph.nodes:
        if node.op == "conv2d":
            consumer = only_consumer(node)
            if consumer is not None and consumer.op == "batchnorm":
                groups.append(("conv_bn", [node.id, consumer.id]))
        elif node.op == "slice":
            shape = graph.tensors[node.inputs[0]].shape
            a = node.attrs
            if (len(shape) != 4 or a.get("axis") != 2 or a.get("start") != 0
                    or a.get("step") != 2 or a.get("end") != shape[2]):
                continue
            second = only_consumer(node)
            if second is None or second.op != "slice":
                continue
            a = second.attrs
            if (a.get("axis") != 3 or a.get("start") != 0 or a.get("step") != 2
                    or a.get("end") != shape[3]):
                continue
            padding = only_consumer(second)
            if padding is None or padding.op != "pad":
                continue
            pads = tuple(padding.attrs.get("pads", ()))
            if len(pads) == 6 and pads[:4] == (0, 0, 0, 0):
                groups.append(("downsample_pad", [node.id, second.id, padding.id]))
        elif (node.op == "mean" and tuple(node.attrs.get("axes", ())) == (2, 3)
              and node.attrs.get("keepdim") is False):
            for consumer in graph.consumers(node.outputs[0]):
                if consumer.op == "linear":
                    groups.append(("mean_head", [node.id, consumer.id]))
    positions = {node.id: index for index, node in enumerate(graph.nodes)}
    groups.sort(key=lambda group: positions[group[1][0]])
    return [dict(pattern=name, group=index, nodes=nodes)
            for index, (name, nodes) in enumerate(groups)]
