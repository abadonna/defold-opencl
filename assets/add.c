__kernel void parallel_add(__global float* x, __global float* y, __global float* z){
	const int i = get_global_id(0); // get a unique number identifying the work item in the global pool
	z[i] = y[i] + x[i]; // add two arrays 
}

__kernel void norm(__global float3* v,  __global float3* res){
	const int i = get_global_id(0); 
	res[i] = normalize(v[i]); 
}
