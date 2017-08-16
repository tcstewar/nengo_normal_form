import pytest

import nengo
import nengo_normal_form
import nengo_benchmarks
import numpy as np

def do_compare(model, time):
    nengo_normal_form.assign_seeds(model)

    sim_standard = nengo.Simulator(model)
    with sim_standard:
        sim_standard.run(time)

    model_n, probes = nengo_normal_form.convert(model)
    sim_normal = nengo.Simulator(model_n)
    with sim_normal:
        sim_normal.run(time)

    for probe, node in probes.items():
        d_s = sim_standard.data[probe]
        d_n = node.data

        assert np.allclose(d_s, d_n, rtol=1e-5)




def test_comm_channel():
    model = nengo_benchmarks.CommunicationChannel().make_model()
    do_compare(model, time=1.0)

def test_convolution():
    model = nengo_benchmarks.CircularConvolution().make_model()
    do_compare(model, time=1.0)

def test_lorenz():
    model = nengo_benchmarks.Lorenz().make_model()
    do_compare(model, time=1.0)

def test_parsing():
    model = nengo_benchmarks.Parsing().make_model()
    do_compare(model, time=1.5)

