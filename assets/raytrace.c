bool rayTriangleIntersect( //scratchapixel.com
	float3 *orig, float3 *dir,
	float3 *v0, float3 *v1, float3 *v2,
	float *t, float *u, float *v)
	{

		const float kEpsilon = 1e-8;

		// compute the plane's normal
		float3 v0v1 = *v1 - *v0;
		float3 v0v2 = *v2 - *v0;
		// no need to normalize
		float3 N = cross(v0v1, v0v2); // N
		float denom = dot(N, N);

		// Step 1: finding P

		// check if the ray and plane are parallel.
		float NdotRayDirection = dot(N, *dir);

		if (fabs(NdotRayDirection) < kEpsilon) // almost 0
		return false; // they are parallel so they don't intersect! 

		// compute d parameter using equation 2
		float d = - dot(N, *v0);


		// compute t (equation 3)
		*t = -(dot(N, *orig) + d) / NdotRayDirection;

		float T = *t;
		// check if the triangle is behind the ray
		if (T < 0) return false; // the triangle is behind

		// compute the intersection point using equation 1
		float3 P = *orig + (float3)(T * dir->x, T * dir->y, T * dir->z);

		// Step 2: inside-outside test
		float3 C; // vector perpendicular to triangle's plane

		// edge 0
		float3 edge0 = *v1 - *v0; 
		float3 vp0 = P - *v0;
		C = cross(edge0, vp0);
		if (dot(N, C) < 0) return false; // P is on the right side


		// edge 1
		float3 edge1 = *v2 - *v1; 
		float3 vp1 = P - *v1;
		C = cross(edge1, vp1);
		if ((*u = dot(N, C)) < 0)  return false; // P is on the right side


		// edge 2
		float3 edge2 = *v0 - *v2; 
		float3 vp2 = P - *v2;
		C = cross(edge2, vp2);
		if ((*v = dot(N, C)) < 0) return false; // P is on the right side;

		*u /= denom;
		*v /= denom;

		return true; // this ray hits the triangle
	}


	int intersect(float3 *orig, float3 *dir, int num_faces, __global float3* p, float *dist, float2 *uv) 
	{
		int face = -1;

		float u, v, t;
		for (int i = 0; i < num_faces; i ++) {
			int k = i * 3;
			float3 v0 = p[k];
			float3 v1 = p[k + 1];
			float3 v2 = p[k + 2];

			if (rayTriangleIntersect(orig, dir, &v0, &v1, &v2, &t, &u, &v) && t < *dist) {
				*dist = t;
				face = i;
				uv->x = u;
				uv->y = v;
			}
		}
		return face;
	}


	float cast_ray(
		float3 *orig, 
		float3 *dir, 
		int num_faces, 
		__global float3 *p, 
		__global float3* n
	) 
	{
		float result = 0.f;
		float dist = FLT_MAX;
		float2 uv;

		int face = intersect(orig, dir, num_faces, p, &dist, &uv);

		if (face > -1) {
			float3 point = (*orig) + (*dir) * dist;
			float3 normal = (1 - uv.x - uv.y) * n[face * 3] + uv.x * n[face * 3 + 1] + uv.y * n[face * 3 + 2];
			normal = normalize(normal);

			float3 light = (float3)(0, 2, 2);
			float3 light_dir = normalize(light - point);

			float diff = max(0.f, dot(normal, light_dir));
			float3 hit = point +  normal * 0.01f; 
			float d = FLT_MAX;
			float2 uv2;

			result = diff;
		}
		return result;
	}


__kernel void trace_scene(
							__global uchar3* color, 
							int width, int height, 
							__global float3* vertices, 
							__global float3* normals, 
							int num_faces)
{
	const int i = get_global_id(0);
	const int j = get_global_id(1);
	
	float aspect = 1.5;
	float tan_hfov = tan(0.7854f * 0.5f);

	float3 p = (float3)(i / (float)( width - 1) - 0.5f, j / (float)(height - 1) - 0.5f, 0) ; 

	p *= tan_hfov;
	p.x *= aspect;
	p.z = 9;

	float3 origin = (float3)(0, 0, 10); //camera position 
	float3 dir = normalize(p - origin);

	float value = cast_ray(&origin, &dir, num_faces, vertices, normals);
	
	color[i  + j * width] =(uchar3)(value * 255, value * 255, value * 255);
}
