import nengo
import numpy as np


class ProbeNode(nengo.Node):
    def __init__(self, size_in):
        super(ProbeNode, self).__init__(output=self.collect,
                                        label='Probe(%d)' % size_in,
                                        size_in=size_in,
                                        size_out=size_in)
        self.data = []

    def collect(self, t, x):
        self.data.append(x)
        return x

def combine_synapses(synapse_in, synapse_out):
    if not isinstance(synapse_in, nengo.synapses.LinearFilter):
        raise Unconvertible("Cannot merge filter %s" % synapse_in)
    if not isinstance(synapse_out, nengo.synapses.LinearFilter):
        raise Unconvertible("Cannot merge filter %s" % synapse_out)

    return nengo.synapses.LinearFilter(
        np.polymul(synapse_in.num, synapse_out.num),
        np.polymul(synapse_in.den, synapse_out.den))


def find_passthrough_nodes(model):
    nodes = [n for n in model.nodes if n.output is None]
    in_nodes = []
    out_nodes = []
    if isinstance(model, nengo.networks.EnsembleArray):
        in_nodes.append(model.input)
        out_nodes = [n for n in nodes if n not in in_nodes]
        nodes = []
    for net in model.networks:
        n, i, o = find_passthrough_nodes(net)
        nodes.extend(n)
        in_nodes.extend(i)
        out_nodes.extend(o)
    return nodes, in_nodes, out_nodes

def create_replacement(c_in, c_out):
    """Generate a new Connection to replace two through a passthrough Node"""
    assert c_in.post_obj is c_out.pre_obj
    assert c_in.post_obj.output is None

    # determine the filter for the new Connection
    if c_in.synapse is None:
        synapse = c_out.synapse
    elif c_out.synapse is None:
        synapse = c_in.synapse
    else:
        synapse = combine_synapses(c_in.synapse, c_out.synapse)

    function = c_in.function
    if c_out.function is not None:
        raise Unconvertible("Cannot remove a connection with a function")

    # compute the combined transform
    transform = np.dot(nengo.utils.builder.full_transform(c_out),
                       nengo.utils.builder.full_transform(c_in))

    # check if the transform is 0 (this happens a lot
    #  with things like identity transforms)
    if np.allclose(transform, 0):
        return None

    if len(transform.shape) == 0:
        c = nengo.Connection(c_in.pre_obj,
                             c_out.post_obj,
                             synapse=synapse,
                             transform=transform,
                             function=function,
                             add_to_container=False)
    else:
        post_indices = []
        for j in range(transform.shape[0]):
            if not np.allclose(transform[j,:], 0):
                post_indices.append(j)
        transform = transform[post_indices, :]
        if function is None:
            pre_indices = []
            for i in range(transform.shape[1]):
                if not np.allclose(transform[:,i], 0):
                    pre_indices.append(i)
            transform = transform[:, pre_indices]
            pre = c_in.pre_obj[pre_indices]
        else:
            pre = c_in.pre

        c = nengo.Connection(pre,
                             c_out.post_obj[post_indices],
                             synapse=synapse,
                             transform=transform,
                             function=function,
                             add_to_container=False)
    return c


def remove_nodes(objs, passthrough, original_conns):
    inputs = {}
    outputs = {}
    for obj in objs:
        inputs[obj] = []
        outputs[obj] = []
    for c in original_conns:
        inputs[c.post_obj].append(c)
        outputs[c.pre_obj].append(c)

    for n in passthrough:
        for c_in in inputs[n]:
            for c_out in outputs[n]:
                c = create_replacement(c_in, c_out)
                if c is not None:
                    outputs[c_in.pre_obj].append(c)
                    inputs[c_out.post_obj].append(c)
        for c_in in inputs[n]:
            outputs[c_in.pre_obj].remove(c_in)
        for c_out in outputs[n]:
            inputs[c_out.post_obj].remove(c_out)
        del inputs[n]
        del outputs[n]

    conns = []
    for cs in inputs.values():
        conns.extend(cs)
    return conns


def convert(model,
            probes_as_nodes=True,
            single_decoder=True,
            fixed_radius=True,
            remove_array_io=True,
            ):
    network = nengo.Network(add_to_container=False)

    # add all Ensembles and Nodes to the Network
    network.ensembles.extend(model.all_ensembles)
    network.nodes.extend(model.all_nodes)

    probes = {}
    probe_conns = []
    if probes_as_nodes:
        for p in model.all_probes:
            with network:
                node = ProbeNode(p.size_in)
            c = nengo.Connection(p.target, node,
                                 synapse=p.synapse,
                                 add_to_container=False)
            probes[p] = node
            probe_conns.append(c)
    else:
        network.probes.extend(model.all_probes)


    conns = model.all_connections + probe_conns

    passthrough, input_nodes, output_nodes = find_passthrough_nodes(model)
    if remove_array_io:
        to_remove = passthrough + input_nodes + output_nodes
    else:
        to_remove = passthrough
    conns = remove_nodes(network.ensembles + network.nodes,
                         to_remove,
                         conns)
    for n in to_remove:
        network.nodes.remove(n)



    for c in conns:
        with network:
            pre = c.pre
            post = c.post
            synapse = c.synapse
            transform = c.transform
            pre_args = dict(solver=c.solver,
                            eval_points=c.eval_points,
                            scale_eval_points=c.scale_eval_points,
                            function = c.function)

            nengo.Connection(pre, post,
                             transform=transform,
                             synapse=synapse,
                             **pre_args)

    return network, probes
