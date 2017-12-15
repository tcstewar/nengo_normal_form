import nengo_normal_form
import nengo
import numpy as np

from .generic import GenericSimulator



class Host2Client(nengo.Node):
    def __init__(self, conn):
        self.latest_time = 0
        self.latest_x = np.zeros(conn.size_out)
        super(Host2Client, self).__init__(self.update,
                                          size_in=conn.post_obj.size_in, 
                                          size_out=0)
        self.post_slice = conn.post_slice
    def update(self, t, x):
        self.latest_time = t
        self.latest_x = x[self.post_slice]

class Client2Host(nengo.Node):
    def __init__(self, conn):
        super(Client2Host, self).__init__(self.update,
                                          size_in=0,
                                          size_out=conn.size_out)
        self.value = np.zeros(conn.size_out)
    def update(self, t):
        return self.value

class HostedSimulator(GenericSimulator):
    def __init__(self, model, dt=0.001, progress_bar=True):
        super(HostedSimulator, self).__init__(dt=dt, progress_bar=progress_bar)

        norm_model, probes = nengo_normal_form.convert(model)

        self.host2client = {}
        self.client2host = {}
        self.client_conns = []
        self.client_objs = []
        host_model = nengo.Network()

        for node in norm_model.nodes:
            if self.is_on_host(node):
                host_model.nodes.append(node)
            else:
                self.client_objs.append(node)
        for ens in norm_model.ensembles:
            if self.is_on_host(ens):
                host_model.ensembles.append(ens)
            else:
                self.client_objs.append(ens)

        for c in norm_model.connections:
            host_pre = self.is_on_host(c.pre_obj)
            host_post = self.is_on_host(c.post_obj)

            if host_pre:
                if host_post:
                    host_model.connections.append(c)
                else:
                    with host_model:
                        self.host2client[c] = Host2Client(c)
                        nengo.Connection(
                                c.pre, 
                                self.host2client[c],
                                synapse=c.synapse,
                                transform=c.transform,
                                function=c.function,
                                label=c.label)
            else:
                if host_post:
                    with host_model:
                        self.client2host[c] = Client2Host(c)
                        nengo.Connection(
                                self.client2host[c],
                                c.post, 
                                synapse=c.synapse,
                                transform=c.transform,
                                label=c.label)
                else:
                    self.client_conns.append(c)


        self.host = nengo.Simulator(host_model, progress_bar=False)

        for p, pnode in probes.items():
            self.data[p] = pnode.data

    def step(self):
        self.host.step()
        super(HostedSimulator, self).step()

    def is_on_host(self, obj):
        if isinstance(obj, nengo_normal_form.DecoderNode):
            return False
        if isinstance(obj, nengo.Node):
            return True
        if isinstance(obj, nengo.Ensemble):
            if isinstance(obj.neuron_type, nengo.Direct):
                return True
            else:
                return False
        raise nengo.exceptions.NengoException(
                'Unhandled connection to/from %s' % obj)



