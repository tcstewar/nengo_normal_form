import nengo
import numpy as np
import copy

from nengo_normal_form import seeds, normal_form, hosted

class ClientSender(nengo.Node):
    def __init__(self, conn):
        self.latest_time = 0
        self.latest_x = np.zeros(conn.size_out)
        super(ClientSender, self).__init__(self.update,
                                          size_in=conn.size_out, 
                                          size_out=0)
    def update(self, t, x):
        self.latest_time = t
        self.latest_x[:] = x

class ClientReceiver(nengo.Node):
    def __init__(self, conn):
        super(ClientReceiver, self).__init__(self.update,
                                          size_in=0,
                                          size_out=conn.size_out)
        self.value = np.zeros(conn.size_out)
    def update(self, t):
        return self.value


class Simulator(hosted.HostedSimulator):
    def __init__(self, model, dt=0.001, progress_bar=True):
        super(Simulator, self).__init__(model, dt=dt, progress_bar=progress_bar)

        generated_seeds = seeds.determine_seeds(model)

        self.client_model = nengo.Network()

        for obj in self.client_objs:
            if isinstance(obj, nengo.Ensemble):
                self.client_model.ensembles.append(obj)
            elif isinstance(obj, normal_form.DecoderNode):
                self.client_model.nodes.append(obj)
            else:
                raise nengo.exceptions.NengoException(
                        'Cannot handle component %s' % obj)

        for conn in self.client_conns:
            self.client_model.connections.append(conn)


        self.receivers = {}
        self.senders = {}
        for conn in self.host2client.keys():
            with self.client_model:
                n = ClientReceiver(conn)
                nengo.Connection(n, conn.post, synapse=None)
                self.receivers[conn] = n
        for conn in self.client2host.keys():
            with self.client_model:
                n = ClientSender(conn)
                nengo.Connection(conn.pre, n, synapse=None)
                self.senders[conn] = n


        old_seeds = {}
        for ens in self.client_model.ensembles:
            old_seeds[ens] = ens.seed
            ens.seed = generated_seeds[ens]
        self.client_sim = nengo.Simulator(self.client_model)
        for ens in self.client_model.ensembles:
            ens.seed = old_seeds[ens]



    def step(self):
        super(Simulator, self).step()
        self.client_sim.step()
        for conn, c_node in self.host2client.items():
            r = self.receivers[conn]
            r.value[:] = c_node.latest_x
        for conn, c_node in self.client2host.items():
            s = self.senders[conn]
            c_node.value[:] = s.latest_x





