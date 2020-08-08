/*
 * linmath_test.h
 *
 *  Created on: Apr 9, 2017
 *      Author: Danylo Ulianych
 */

#ifndef LINMATH_TEST_H
#define LINMATH_TEST_H

#include <stdlib.h>
#include "linmath.h"

#define LINMATH_EPS                   (0.0001f)
#define linmath_is_close(val1, val2)  (fabsf(val1 - val2) < LINMATH_EPS)

#ifndef linmath_assert
#include <assert.h>
#define linmath_assert                assert
#endif /* linmath_assert */


static float linmath_random_float() {
	return rand() / (float) RAND_MAX;
}

#define LINMATH_TEST_DEFINE_VEC(n) \
static void linmath_vec##n##_set(vec##n v, float value) { \
	int i; \
	for (i=0; i<n; i++) { \
		v[i] = value; \
	} \
} \
static void linmath_vec##n##_init_random(vec##n v) { \
	int i; \
	for (i=0; i<n; i++) { \
		v[i] = linmath_random_float(); \
	} \
} \
static int linmath_vec##n##_allclose(vec##n const a, vec##n const b) { \
	int i, equal = 1; \
	for(i = 0; i < n; ++i) \
		equal &= linmath_is_close(a[i], b[i]); \
	return equal; \
} \
static void linmath_test_vec##n##_mul_inner() { \
	/* The inner product of a vector of ones with itself must equal 'n'. */ \
	vec##n v; \
	linmath_vec##n##_set(v, 1.0f); \
	float inner_prod = vec##n##_mul_inner(v, v); \
	linmath_assert(linmath_is_close(inner_prod, n)); \
} \
static void linmath_test_vec##n##_len() { \
	/* The length of a vector of ones must equal sqrt(n). */ \
	vec##n v; \
	int i; \
	for (i=0; i<n; i++) { \
		v[i] = 1.0f; \
	} \
	float norm = vec##n##_len(v); \
	linmath_assert(linmath_is_close(norm, sqrtf(n))); \
} \
static void linmath_test_vec##n##_norm() { \
	/* The norm of a normalized vector must be 1.0. */ \
	srand(17U);  /* set any seed */ \
	vec##n v; \
	linmath_vec##n##_init_random(v); \
	vec##n r; \
	vec##n##_norm(r, v); \
	float norm = vec##n##_len(r); \
	linmath_assert(linmath_is_close(norm, 1.0f)); \
}


LINMATH_TEST_DEFINE_VEC(2);
LINMATH_TEST_DEFINE_VEC(3);
LINMATH_TEST_DEFINE_VEC(4);


static void linmath_test_vec3_mul_cross() {
	srand(13U);  /* set any seed */
	vec3 v1, v2, r;
	linmath_vec3_init_random(v1);
	vec3_dup(v2, v1);
	vec3_mul_cross(r, v1, v2);
	vec3 v_expected;

	// the cross product of equal vectors must be zero
	linmath_vec3_set(v_expected, 0.0f);
	linmath_assert(linmath_vec3_allclose(r, v_expected));

	// test ijk axes cross product
	vec3 i = {1, 0, 0};
	vec3 j = {0, 1, 0};
	vec3 k = {0, 0, 1};
	vec3_mul_cross(r, i, j);
	linmath_assert(linmath_vec3_allclose(r, k));
}

static void linmath_test_vec4_mul_cross() {
	srand(13U);  /* set any seed */
	vec4 v1, v2, r;
	linmath_vec4_init_random(v1);
	vec4_dup(v2, v1);
	vec4_mul_cross(r, v1, v2);
	vec4 v_expected;

	// the cross product of equal vectors must be zero
	linmath_vec4_set(v_expected, 0.0f);
	v_expected[3] = 1.0f;
	linmath_assert(linmath_vec4_allclose(r, v_expected));

	// test ijk axes cross product
	vec4 i = {1, 0, 0, 1};
	vec4 j = {0, 1, 0, 1};
	vec4 k = {0, 0, 1, 1};
	vec4_mul_cross(r, i, j);
	linmath_assert(linmath_vec4_allclose(r, k));
}


static int linmath_mat4x4_allclose(mat4x4 const M, mat4x4 const N) {
	int i, equal = 1;
	for (i = 0; i < 4; ++i)
		equal &= linmath_vec4_allclose(M[i], N[i]);
	return equal;
}

/**
 * Test the correctnes of a quaternion creation
 * that is used as a linear operator to rotate
 * vectors and matrices (later on).
 */
static void linmath_test_quat_rotate() {
    vec3 axis = {0, 1, 0};
	quat q;
	float theta = M_PI_4;
	quat_rotate(q, theta, axis);
	quat q_refernce = {0, sinf(theta / 2), 0, cosf(theta / 2)};
	linmath_assert(linmath_vec4_allclose(q, q_refernce));
}

/**
 * The conjugate of a quaternion must correspond to
 * the rotation with a negative angle.
 */
static void linmath_test_quat_conj() {
    srand(15U);
	quat q, q_conj, q_reference;
	vec3 axis = { 0, 1, 0 };
	float angle_rads = linmath_random_float();
	quat_rotate(q, angle_rads, axis);
	quat_conj(q_conj, q);
	quat_rotate(q_reference, -angle_rads, axis);
	linmath_assert(linmath_vec4_allclose(q_conj, q_reference));
}

/* Rotate a vector back and forth. */
static void linmath_test_quat_mul_vec3() {
	srand(11U);
	quat q, q_conj;
	vec3 axis = { 0, 1, 0 };
	float angle_rads = linmath_random_float();
	quat_rotate(q, angle_rads, axis);
	quat_conj(q_conj, q);

	vec3 v_initial, v_rotated, v_restored;
	linmath_vec3_init_random(v_initial);
	quat_mul_vec3(v_rotated, q, v_initial);
	quat_mul_vec3(v_restored, q_conj, v_rotated);
	linmath_assert(linmath_vec3_allclose(v_restored, v_initial));
}

/* Rotate a matrix back and forth. */
static void linmath_test_mat4x4o_mul_quat() {
	srand(12U);
	quat q, q_conj;
	vec3 axis = { 0, 1, 0 };
	float angle_rads = linmath_random_float();
	quat_rotate(q, angle_rads, axis);
	quat_conj(q_conj, q);

	mat4x4 m_reference, m_rotated, m;
	mat4x4_identity(m_reference);
	m_reference[0][3] = 0.1f;
	m_reference[1][3] = 0.2f;
	m_reference[2][3] = 0.3f;
	mat4x4o_mul_quat(m_rotated, m_reference, q);
	mat4x4o_mul_quat(m, m_rotated, q_conj);
	linmath_assert(linmath_mat4x4_allclose(m_reference, m));
}

/**
 * Test if an extracted quaternion from from a
 * rotational matrix matches the original one.
 */
static void linmath_test_quat_from_mat4x4() {
	srand(7U);
	quat q_reference;
	vec3 axis = { 0, 1, 0 };
	float angle_rads = linmath_random_float();
	quat_rotate(q_reference, angle_rads, axis);

	mat4x4 m_identity, m_rotated;
	mat4x4_identity(m_identity);
	mat4x4o_mul_quat(m_rotated, m_identity, q_reference);

	quat q_restored;
	quat_from_mat4x4(q_restored, m_rotated);
	linmath_assert(linmath_vec4_allclose(q_restored, q_reference));
}



static void linmath_test_run_all() {

	linmath_test_vec2_mul_inner();
	linmath_test_vec3_mul_inner();
	linmath_test_vec4_mul_inner();

	linmath_test_vec2_len();
	linmath_test_vec3_len();
	linmath_test_vec4_len();

	linmath_test_vec2_norm();
	linmath_test_vec3_norm();
	linmath_test_vec4_norm();

	linmath_test_vec3_mul_cross();
	linmath_test_vec4_mul_cross();

	linmath_test_quat_rotate();
	linmath_test_quat_conj();
	linmath_test_quat_mul_vec3();
	linmath_test_mat4x4o_mul_quat();

	/* FIXME: Below is the wrecked functional that does not work */
    // linmath_test_quat_from_mat4x4();
}


#endif /* LINMATH_TEST_H */
