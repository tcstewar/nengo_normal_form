import nengo_benchmarks



model = nengo_benchmarks.CircularConvolution().make_model(D=3)

import nengo_normal_form

import imp
imp.reload(nengo_normal_form)
imp.reload(nengo_normal_form.normal_form)

model, probes = nengo_normal_form.normal_form.convert(model, 
                    remove_array_io=True,
                    probes_as_nodes=True)
                    
for k, v in probes.items():
    print(k, v)