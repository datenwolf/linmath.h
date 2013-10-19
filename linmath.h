#ifndef LINMATH_H
#define LINMATH_H

#include <math.h>
#include <string.h>

typedef float vec3[3];
static inline void vec3_add(vec3 r, vec3 a, vec3 b)
{
	int i;
	for(i=0; i<3; ++i)
		r[i] = a[i] + b[i];
}
static inline void vec3_sub(vec3 r, vec3 a, vec3 b)
{
	int i;
	for(i=0; i<3; ++i)
		r[i] = a[i] - b[i];
}
static inline void vec3_scale(vec3 r, vec3 v, float s)
{
	int i;
	for(i=0; i<3; ++i)
		r[i] = v[i] * s;
}
static inline float vec3_mul_inner(vec3 a, vec3 b)
{
	float p = 0.;
	int i;
	for(i=0; i<3; ++i)
		p += b[i]*a[i];
	return p;
}
static inline void vec3_mul_cross(vec3 r, vec3 a, vec3 b)
{
	vec3 c;
	c[0] = a[1]*b[2] - a[2]*b[1];
	c[1] = a[2]*b[0] - a[0]*b[2];
	c[2] = a[0]*b[1] - a[1]*b[0];
	memcpy(r, c, sizeof(c));
}
static inline float vec3_len(vec3 v)
{
	return sqrtf(vec3_mul_inner(v,v));
}
static inline void vec3_norm(vec3 r, vec3 v)
{
	float k = 1.0 / vec3_len(v);
	vec3_scale(r, v, k);
}

typedef float vec4[4];
static inline void vec4_add(vec4 r, vec4 a, vec4 b)
{
	int i;
	for(i=0; i<4; ++i)
		r[i] = a[i] + b[i];
}
static inline void vec4_sub(vec4 r, vec4 a, vec4 b)
{
	int i;
	for(i=0; i<4; ++i)
		r[i] = a[i] - b[i];
}
static inline void vec4_scale(vec4 r, vec4 v, float s)
{
	int i;
	for(i=0; i<4; ++i)
		r[i] = v[i] * s;
}
static inline float vec4_mul_inner(vec4 a, vec4 b)
{
	float p = 0.;
	int i;
	for(i=0; i<4; ++i)
		p += b[i]*a[i];
	return p;
}
static inline void vec4_mul_cross(vec4 r, vec4 a, vec4 b)
{
	vec4 c;
	c[0] = a[1]*b[2] - a[2]*b[1];
	c[1] = a[2]*b[0] - a[0]*b[2];
	c[2] = a[0]*b[1] - a[1]*b[0];
	c[3] = 1.;
	memcpy(r, c, sizeof(c));
}
static inline float vec4_len(vec4 v)
{
	return sqrtf(vec4_mul_inner(v,v));
}
static inline void vec4_norm(vec4 r, vec4 v)
{
	float k = 1.0 / vec4_len(v);
	vec4_scale(r, v, k);
}

typedef vec4 mat4x4[4];
static inline void mat4x4_identity(mat4x4 M)
{
	int i, j;
	for(j=0; j<4; ++j) for(i=0; i<4; ++i) {
		M[i][j] = i==j ? 1 : 0;
	}
}
static inline void mat4x4_dup(mat4x4 M, mat4x4 N)
{
	int i, j;
	for(j=0; j<4; ++j) {
		for(i=0; i<4; ++i) {
			M[i][j] = N[i][j];
		}
	}
}
static inline void mat4x4_row(vec4 r, mat4x4 M, int i)
{
	int k;
	for(k=0; k<4; ++k)
		r[k] = M[k][i];
}
static inline void mat4x4_col(vec4 r, mat4x4 M, int i)
{
	int k;
	for(k=0; k<4; ++k)
		r[k] = M[i][k];
}
static inline void mat4x4_transpose(mat4x4 M, mat4x4 N)
{
	int i, j;
	mat4x4 R;
	for(j=0; j<4; ++j) {
		for(i=0; i<4; ++i) {
			R[i][j] = N[j][i];
		}
	}
	memcpy(M, R, sizeof(R));
}
static inline void mat4x4_add(mat4x4 M, mat4x4 a, mat4x4 b)
{
	int i;
	for(i=0; i<4; ++i)
		vec4_add(M[i], a[i], b[i]);
}
static inline void mat4x4_sub(mat4x4 M, mat4x4 a, mat4x4 b)
{
	int i;
	for(i=0; i<4; ++i)
		vec4_sub(M[i], a[i], b[i]);
}
static inline void mat4x4_scale(mat4x4 M, mat4x4 a, float k)
{
	int i;
	for(i=0; i<4; ++i)
		vec4_scale(M[i], a[i], k);
}
static inline void mat4x4_scale_aniso(mat4x4 M, mat4x4 a, float x, float y, float z)
{
	vec4_scale(M[0], a[0], x);
	vec4_scale(M[1], a[1], y);
	vec4_scale(M[2], a[2], z);
}
static inline void mat4x4_mul(mat4x4 M, mat4x4 a, mat4x4 b)
{
	int k, r, c;
	mat4x4 R;
	for(r=0; r<4; ++r) for(c=0; c<4; ++c) {
		R[c][r] = 0;
		for(k=0; k<4; ++k) {
			R[c][r] += a[k][r] * b[c][k];
		}
	}
	memcpy(M, R, sizeof(R));
}
static inline void mat4x4_mul_vec4(vec4 r, mat4x4 M, vec4 v)
{
	vec4 r_;
	int i, j;
	for(j=0; j<4; ++j) {
		r_[j] = 0.;
		for(i=0; i<4; ++i) {
			r_[j] += M[i][j] * v[i];
		}
	}
	memcpy(r, r_, sizeof(r_));
}
static inline void mat4x4_translate(mat4x4 T, float x, float y, float z)
{
	mat4x4_identity(T);
	T[3][0] = x;
	T[3][1] = y;
	T[3][2] = z;
}
static inline void mat4x4_translate_in_place(mat4x4 M, float x, float y, float z)
{
	vec4 t = {x, y, z, 1};
	vec4 r;
	int i;
	for (i = 0; i < 4; ++i) {
		mat4x4_row(r, M, i);
		M[3][i] += vec4_mul_inner(r, t);
	}
}
static inline void mat4x4_from_vec3_mul_outer(mat4x4 M, vec3 a, vec3 b)
{
	int i, j;
	for(i=0; i<4; ++i) for(j=0; j<4; ++j) {
		M[i][j] = i<3 && j<3 ? a[i] * b[j] : 0.;
	}
}
static inline void mat4x4_rotate(mat4x4 R, mat4x4 M, float x, float y, float z, float angle)
{
	float s = sinf(angle);
	float c = cosf(angle);
	vec3 u = {x, y, z};
	vec3_norm(u, u);

	{
		mat4x4 T;
		mat4x4_from_vec3_mul_outer(T, u, u);

		mat4x4 S = {
			{    0,  u[2], -u[1], 0},
			{-u[2],     0,  u[0], 0},
			{ u[1], -u[0],     0, 0},
			{    0,     0,     0, 0}
		};
		mat4x4_scale(S, S, s);

		mat4x4 C;
		mat4x4_identity(C);
		mat4x4_sub(C, C, T);

		mat4x4_scale(C, C, c);

		mat4x4_add(T, T, C);
		mat4x4_add(T, T, S);

		T[3][3] = 1.;		
		mat4x4_mul(R, M, T);
	}
}
static inline void mat4x4_rotate_X(mat4x4 Q, mat4x4 M, float angle)
{
	float s = sinf(angle);
	float c = cosf(angle);
	mat4x4 R = {
		{1, 0, 0, 0},
		{0, c, s, 0},
		{0,-s, c, 0},
		{0, 0, 0, 1}
	};
	mat4x4_mul(Q, M, R);
}
static inline void mat4x4_rotate_Y(mat4x4 Q, mat4x4 M, float angle)
{
	float s = sinf(angle);
	float c = cosf(angle);
	mat4x4 R = {
		{ c, 0, s, 0},
		{ 0, 1, 0, 0},
		{-s, 0, c, 0},
		{ 0, 0, 0, 1}
	};
	mat4x4_mul(Q, M, R);
}
static inline void mat4x4_rotate_Z(mat4x4 Q, mat4x4 M, float angle)
{
	float s = sinf(angle);
	float c = cosf(angle);
	mat4x4 R = {
		{ c, s, 0, 0},
		{-s, c, 0, 0},
		{ 0, 0, 1, 0},
		{ 0, 0, 0, 1}
	};
	mat4x4_mul(Q, M, R);
}
static inline void mat4x4_invert(mat4x4 T, mat4x4 M)
{
	mat4x4 R;
	R[0][0] = M[1][1]*(M[2][2]*M[3][3] - M[2][3]*M[3][2]) - M[2][1]*(M[1][2]*M[3][3] - M[1][3]*M[3][2]) - M[3][1]*(M[1][3]*M[2][2] - M[1][2]*M[2][3]);
	R[0][1] = M[0][1]*(M[2][3]*M[3][2] - M[2][2]*M[3][3]) - M[2][1]*(M[0][3]*M[3][2] - M[0][2]*M[3][3]) - M[3][1]*(M[0][2]*M[2][3] - M[0][3]*M[2][2]);
	R[0][2] = M[0][1]*(M[1][2]*M[3][3] - M[1][3]*M[3][2]) - M[1][1]*(M[0][2]*M[3][3] - M[0][3]*M[3][2]) - M[3][1]*(M[0][3]*M[1][2] - M[0][2]*M[1][3]);
	R[0][3] = M[0][1]*(M[1][3]*M[2][2] - M[1][2]*M[2][3]) - M[1][1]*(M[0][3]*M[2][2] - M[0][2]*M[2][3]) - M[2][1]*(M[0][2]*M[1][3] - M[0][3]*M[1][2]);

	R[1][0] = M[1][0]*(M[2][3]*M[3][2] - M[2][2]*M[3][3]) - M[2][0]*(M[1][3]*M[3][2] - M[1][2]*M[3][3]) - M[3][0]*(M[1][2]*M[2][3] - M[1][3]*M[2][2]);
	R[1][1] = M[0][0]*(M[2][2]*M[3][3] - M[2][3]*M[3][2]) - M[2][0]*(M[0][2]*M[3][3] - M[0][3]*M[3][2]) - M[3][0]*(M[0][3]*M[2][2] - M[0][2]*M[2][3]);
	R[1][2] = M[0][0]*(M[1][3]*M[3][2] - M[1][2]*M[3][3]) - M[1][0]*(M[0][3]*M[3][2] - M[0][2]*M[3][3]) - M[3][0]*(M[0][2]*M[1][3] - M[0][3]*M[1][2]);
	R[1][3] = M[0][0]*(M[1][2]*M[2][3] - M[1][3]*M[2][2]) - M[1][0]*(M[0][2]*M[2][3] - M[0][3]*M[2][2]) - M[2][0]*(M[0][3]*M[1][2] - M[0][2]*M[1][3]);

	R[2][0]  = M[1][0]*(M[2][1]*M[3][3] - M[2][3]*M[3][1]) - M[2][0]*(M[1][1]*M[3][3] - M[1][3]*M[3][1]) - M[3][0]*(M[1][3]*M[2][1] - M[1][1]*M[2][3]);
	R[2][1]  = M[0][0]*(M[2][3]*M[3][1] - M[2][1]*M[3][3]) - M[2][0]*(M[0][3]*M[3][1] - M[0][1]*M[3][3]) - M[3][0]*(M[0][1]*M[2][3] - M[0][3]*M[2][1]);
	R[2][2] = M[0][0]*(M[1][1]*M[3][3] - M[1][3]*M[3][1]) - M[1][0]*(M[0][1]*M[3][3] - M[0][3]*M[3][1]) - M[3][0]*(M[0][3]*M[1][1] - M[0][1]*M[1][3]);
	R[2][3] = M[0][0]*(M[1][3]*M[2][1] - M[1][1]*M[2][3]) - M[1][0]*(M[0][3]*M[2][1] - M[0][1]*M[2][3]) - M[2][0]*(M[0][1]*M[1][3] - M[0][3]*M[1][1]);

	R[3][0] = M[1][0]*(M[2][2]*M[3][1] - M[2][1]*M[3][2]) - M[2][0]*(M[1][2]*M[3][1] - M[1][1]*M[3][2]) - M[3][0]*(M[1][1]*M[2][2] - M[1][2]*M[2][1]);
	R[3][1] = M[0][0]*(M[2][1]*M[3][2] - M[2][2]*M[3][1]) - M[2][0]*(M[0][1]*M[3][2] - M[0][2]*M[3][1]) - M[3][0]*(M[0][2]*M[2][1] - M[0][1]*M[2][2]);
	R[3][2] = M[0][0]*(M[1][2]*M[3][1] - M[1][1]*M[3][2]) - M[1][0]*(M[0][2]*M[3][1] - M[0][1]*M[3][2]) - M[3][0]*(M[0][1]*M[1][2] - M[0][2]*M[1][1]);
	R[3][3] = M[0][0]*(M[1][1]*M[2][2] - M[1][2]*M[2][1]) - M[1][0]*(M[0][1]*M[2][2] - M[0][2]*M[2][1]) - M[2][0]*(M[0][2]*M[1][1] - M[0][1]*M[1][2]);
	memcpy(T, R, sizeof(T));
}
static inline void mat4x4_frustum(mat4x4 M, float l, float r, float b, float t, float n, float f)
{
	M[0][0] = 2.*n/(r-l);
	M[0][1] = M[0][2] = M[0][3] = 0.;
	
	M[1][1] = 2.*n/(t-b);
	M[1][0] = M[1][2] = M[1][3] = 0.;

	M[2][0] = (r+l)/(r-l);
	M[2][1] = (t+b)/(t-b);
	M[2][2] = -(f+n)/(f-n);
	M[2][3] = -1;
	
	M[3][2] = -2.*(f*n)/(f-n);
	M[3][0] = M[3][1] = M[3][3] = 0.;
}
static inline void mat4x4_ortho(mat4x4 M, float l, float r, float b, float t, float n, float f)
{
	M[0][0] = 2./(r-l);
	M[0][1] = M[0][2] = M[0][3] = 0.;

	M[1][1] = 2./(t-b);
	M[1][0] = M[1][2] = M[1][3] = 0.;

	M[2][2] = -2./(f-n);
	M[2][0] = M[2][1] = M[2][3] = 0.;
	
	M[3][0] = -(r+l)/(r-l);
	M[3][1] = -(t+b)/(t-b);
	M[3][2] = -(f+n)/(f-n);
	M[3][3] = 1.;
}
static inline void mat4x4_perspective(mat4x4 m, float y_fov_in_degrees, float aspect, float n, float f)
{
	/* Adapted from Android's OpenGL Matrix.java. */
	float const angle_in_radians = (float) (y_fov_in_degrees * M_PI / 180.0);
	float const a = (float) (1.0 / tan(angle_in_radians / 2.0));

	m[0][0] = a / aspect;
	m[1][0] = 0.0f;
	m[2][0] = 0.0f;
	m[3][0] = 0.0f;

	m[1][0] = 0.0f;
	m[1][1] = a;
	m[1][2] = 0.0f;
	m[1][3] = 0.0f;

	m[2][0] = 0.0f;
	m[2][1] = 0.0f;
	m[2][2] = -((f + n) / (f - n));
	m[2][3] = -1.0f;

	m[3][0] = 0.0f;
	m[3][1] = 0.0f;
	m[3][2] = -((2.0f * f * n) / (f - n));
	m[3][3] = 0.0f;
}
static inline void mat4x4_look_at(mat4x4 m, vec3 eye, vec3 center, vec3 up)
{
	/* Adapted from Android's OpenGL Matrix.java. */
	// See the OpenGL GLUT documentation for gluLookAt for a description
	// of the algorithm. We implement it in a straightforward way:
	vec3 f;
	vec3_sub(f, center, eye);	
	vec3_norm(f, f);	
	
	vec3 s;
	vec3_mul_cross(s, f, up);
	vec3_norm(s, s);

	vec3 u;
	vec3_mul_cross(u, s, f);

	m[0][0] = s[0];
	m[0][1] = u[0];
	m[0][2] = -f[0];
	m[0][3] = 0.0f;

	m[1][0] = s[1];
	m[1][1] = u[1];
	m[1][2] = -f[1];
	m[1][3] = 0.0f;

	m[2][0] = s[2];
	m[2][1] = u[2];
	m[2][2] = -f[2];
	m[2][3] = 0.0f;

	m[3][0] = 0.0f;
	m[3][1] = 0.0f;
	m[3][2] = 0.0f;
	m[3][3] = 1.0f;

	mat4x4_translate_in_place(m, -eye[0], -eye[1], -eye[2]);
}

typedef float quat[4];
static inline void quat_identity(quat q)
{
	q[0] = q[1] = q[2] = 0.;
	q[3] = 1.;
}
static inline void quat_add(quat r, quat a, quat b)
{
	int i;
	for(i=0; i<4; ++i)
		r[i] = a[i] + b[i];
}
static inline void quat_sub(quat r, quat a, quat b)
{
	int i;
	for(i=0; i<4; ++i)
		r[i] = a[i] - b[i];
}
static inline void quat_mul(quat r, quat p, quat q)
{
	vec3 w;
	vec3_mul_cross(r, p, q);
	vec3_scale(w, p, q[3]);
	vec3_add(r, r, w);
	vec3_scale(w, q, p[3]);
	vec3_add(r, r, w);
	r[3] = p[3]*q[3] - vec3_mul_inner(p, q);
}
static inline void quat_scale(quat r, quat v, float s)
{
	int i;
	for(i=0; i<4; ++i)
		r[i] = v[i] * s;
}
static inline float quat_inner_product(quat a, quat b)
{
	float p = 0.;
	int i;
	for(i=0; i<4; ++i)
		p += b[i]*a[i];
	return p;
}
static inline void quat_conj(quat r, quat q)
{
	int i;
	for(i=0; i<3; ++i)
		r[i] = -q[i];
	r[3] = q[3];
}
#define quat_norm vec4_norm
static inline void quat_mul_vec3(vec3 r, quat q, vec3 v)
{
	quat q_;
	quat v_ = {v[0], v[1], v[2], 0.};

	quat_conj(q_, q);
	quat_norm(q_, q_);
	quat_mul(q_, v_, q_);
	quat_mul(q_, q, q_);
	memcpy(r, q_, sizeof(r));
}
static inline void mat4x4_from_quat(mat4x4 M, quat q)
{
	float a = q[3];
	float b = q[0];
	float c = q[1];
	float d = q[2];
	float a2 = a*a;
	float b2 = b*b;
	float c2 = c*c;
	float d2 = d*d;
	
	M[0][0] = a2 + b2 - c2 - d2;
	M[0][1] = 2*(b*c + a*d);
	M[0][2] = 2*(b*d - a*c);
	M[0][3] = 0.;

	M[1][0] = 2*(b*c - a*d);
	M[1][1] = a2 - b2 + c2 - d2;
	M[1][2] = 2*(c*d + a*b);
	M[1][3] = 0.;

	M[2][0] = 2*(b*d + a*c);
	M[2][1] = 2*(c*d - a*b);
	M[2][2] = a2 - b2 - c2 + d2;
	M[2][3] = 0.;

	M[3][0] = M[3][1] = M[3][2] = 0.;
	M[3][3] = 1.;
}
static inline void mat4x4_mul_quat(mat4x4 R, mat4x4 M, quat q)
{
	quat_mul_vec3(R[0], M[0], q);
	quat_mul_vec3(R[1], M[1], q);
	quat_mul_vec3(R[2], M[2], q);

	R[3][0] = R[3][1] = R[3][2] = 0.;
	R[3][3] = 1.;
}
static inline void quat_from_mat4x4(quat q, mat4x4 M)
{
	float r=0.;
	int i;

	int perm[] = { 0, 1, 2, 0, 1 };
	int *p = perm;

	for(i = 0; i<3; i++) {
		float m = M[i][i];
		if( m < r )
			continue;
		m = r;
		p = &perm[i];
	}

	r = sqrtf(1. + M[p[0]][p[0]] - M[p[1]][p[1]] - M[p[2]][p[2]] );

	if(r < 1e-6) {
		q[0] = 1.;
		q[1] = q[2] = q[3] = 0.;
		return;
	}

	q[0] = r/2.;
	q[1] = (M[p[0]][p[1]] - M[p[1]][p[0]])/(2.*r);
	q[2] = (M[p[2]][p[0]] - M[p[0]][p[2]])/(2.*r);
	q[3] = (M[p[2]][p[1]] - M[p[1]][p[2]])/(2.*r);
}

#endif
