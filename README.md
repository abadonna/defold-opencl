# Defold-OpenCL
Proof of concept: OpenCL as native extension for Defold.

## Installation
Open your game.project file and add in the dependencies:

```
https://github.com/abadonna/defold-opencl/archive/master.zip
```

## Documentation
I tried to put most boilerplate code under the hood to simplify the use. But also it means less flexibility - can be solved by expanding the libray.

First you need to select device either from all or only GPU devices. To retrive list of available devices call:

```
devices = opencl.get_devices(is_gpu_only)
```

Second step is to build program from code:

```
program = device:load_program(src)
```

It's convinient to keep OpenCL code in custom resources and load it with sys.load_resource(path)

Now you have to create kernel from compiled program code:

```
kernel = program:create_kernel(name)
```

And set all the nessesary arguments, with either:

```
kernel:set_arg_buffer(index, buffer, stream_name, read, write) -- set array as argument at index from dmBuffer
kernel:set_arg_int(index, value) -- set integer
kernel:set_arg_float(index, value) -- set float
kernel:set_arg_vec3(index, value) -- set vmath.vector3
kernel:set_arg_null(index, size) -- set null for local memory buffer
```

Now we can run kernel with

```
time = kernel:run(dimensions, {global_items_in_dimension1, ...}, {local_items_in_dimension1, ...})
```

Local items size can be omitted. Number of work groups will be created = global_items/local_items.
Finally we need to read data back to lua.

```
result_table = kernel:read(index, count) -- read count items from kernel arguments at index and returns as table
kernel:read(index, count, buffer, stream_name) -- read count items from kernel arguments at index to dmBuffer (faster)
```

For more advanced examples check https://github.com/abadonna/defold-light-probes/tree/opencl

