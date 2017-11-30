import numpy as np
import nengo.utils.numpy as npext


def assign_seeds(net, rng=np.random):
    if net.seed is None:
        # this only happens at the very top level
        net.seed = rng.randint(npext.maxint)

    rng = np.random.RandomState(seed=net.seed)

    # let's use the same algorithm as the builder, just to be consistent
    sorted_types = sorted(net.objects, key=lambda t: t.__name__)
    for obj_type in sorted_types:
        for obj in net.objects[obj_type]:
            # generate a seed for each item, so that manually setting a seed
            #  for a particular item doesn't change the generated seed for
            #  other items
            generated_seed = rng.randint(npext.maxint)
            if obj.seed is None:
                obj.seed = generated_seed
    for subnet in net.networks:
        assign_seeds(subnet)

def determine_seeds(net, rng=np.random, seeds=None):
    if seeds is None:
        seeds = {}

    if net.seed is None:
        # this only happens at the very top level
        seed = rng.randint(npext.maxint)
        seeds[net] = seed
    else:
        seeds[net] = net.seed

    rng = np.random.RandomState(seed=seeds[net])

    # let's use the same algorithm as the builder, just to be consistent
    sorted_types = sorted(net.objects, key=lambda t: t.__name__)
    for obj_type in sorted_types:
        for obj in net.objects[obj_type]:
            # generate a seed for each item, so that manually setting a seed
            #  for a particular item doesn't change the generated seed for
            #  other items
            generated_seed = rng.randint(npext.maxint)
            if obj.seed is None:
                seeds[obj] = generated_seed
            else:
                seeds[obj] = obj.seed
    for subnet in net.networks:
        determine_seeds(subnet, seeds=seeds)

    return seeds

