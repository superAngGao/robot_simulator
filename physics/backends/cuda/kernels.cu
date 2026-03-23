/*
 * CUDA kernels for batched robot simulation.
 *
 * Each kernel launches with N threads (one per environment).
 * Within each thread, sequential loops over the body tree (~17 bodies).
 *
 * Spatial vector convention: [linear(3); angular(3)]
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Joint type constants (match static_data.py)
#define JOINT_FREE 0
#define JOINT_REVOLUTE 1
#define JOINT_PRISMATIC 2
#define JOINT_FIXED 3

// Max bodies per robot (compile-time bound for stack arrays)
#define MAX_BODIES 32

// ─────────────────────────────────────────────────────────────────────────────
// Inline device helpers
// ─────────────────────────────────────────────────────────────────────────────

// 3x3 matrix stored as float[9] in row-major
struct Mat33 { float m[9]; };
struct Vec3 { float x, y, z; };
struct Vec6 { float v[6]; };
struct Mat66 { float m[36]; };

__device__ __forceinline__ float mat33_get(const Mat33& M, int r, int c) {
    return M.m[r * 3 + c];
}
__device__ __forceinline__ void mat33_set(Mat33& M, int r, int c, float val) {
    M.m[r * 3 + c] = val;
}

__device__ Mat33 mat33_identity() {
    Mat33 R;
    for (int i = 0; i < 9; i++) R.m[i] = 0.f;
    R.m[0] = R.m[4] = R.m[8] = 1.f;
    return R;
}

__device__ Mat33 mat33_mul(const Mat33& A, const Mat33& B) {
    Mat33 C;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            float s = 0.f;
            for (int k = 0; k < 3; k++)
                s += mat33_get(A, i, k) * mat33_get(B, k, j);
            mat33_set(C, i, j, s);
        }
    return C;
}

__device__ Vec3 mat33_vec3_mul(const Mat33& A, Vec3 v) {
    Vec3 r;
    r.x = A.m[0]*v.x + A.m[1]*v.y + A.m[2]*v.z;
    r.y = A.m[3]*v.x + A.m[4]*v.y + A.m[5]*v.z;
    r.z = A.m[6]*v.x + A.m[7]*v.y + A.m[8]*v.z;
    return r;
}

__device__ Mat33 mat33_transpose(const Mat33& A) {
    Mat33 T;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            mat33_set(T, i, j, mat33_get(A, j, i));
    return T;
}

__device__ Vec3 cross3(Vec3 a, Vec3 b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}

__device__ Vec3 vec3_add(Vec3 a, Vec3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
__device__ Vec3 vec3_sub(Vec3 a, Vec3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
__device__ Vec3 vec3_scale(Vec3 a, float s) { return {a.x*s, a.y*s, a.z*s}; }
__device__ Vec3 vec3_neg(Vec3 a) { return {-a.x, -a.y, -a.z}; }

// Rodrigues rotation
__device__ Mat33 rodrigues(Vec3 axis, float angle) {
    float c = cosf(angle), s = sinf(angle), omc = 1.f - c;
    Mat33 R;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            float eye = (i == j) ? 1.f : 0.f;
            // skew
            float skew = 0.f;
            if (i==0 && j==1) skew = -axis.z;
            if (i==0 && j==2) skew = axis.y;
            if (i==1 && j==0) skew = axis.z;
            if (i==1 && j==2) skew = -axis.x;
            if (i==2 && j==0) skew = -axis.y;
            if (i==2 && j==1) skew = axis.x;
            float ax_i = (i==0)?axis.x:(i==1)?axis.y:axis.z;
            float ax_j = (j==0)?axis.x:(j==1)?axis.y:axis.z;
            mat33_set(R, i, j, c*eye + s*skew + omc*ax_i*ax_j);
        }
    return R;
}

// Quaternion to rotation
__device__ Mat33 quat_to_rot(float qw, float qx, float qy, float qz) {
    float n = sqrtf(qw*qw + qx*qx + qy*qy + qz*qz);
    float w=qw/n, x=qx/n, y=qy/n, z=qz/n;
    Mat33 R;
    R.m[0]=1-2*(y*y+z*z); R.m[1]=2*(x*y-w*z); R.m[2]=2*(x*z+w*y);
    R.m[3]=2*(x*y+w*z); R.m[4]=1-2*(x*x+z*z); R.m[5]=2*(y*z-w*x);
    R.m[6]=2*(x*z-w*y); R.m[7]=2*(y*z+w*x); R.m[8]=1-2*(x*x+y*y);
    return R;
}

// Plücker velocity transform: parent -> child
__device__ Vec6 transform_velocity(const Mat33& R, Vec3 r, Vec6 v) {
    Mat33 E = mat33_transpose(R);
    Vec3 vl = {v.v[0], v.v[1], v.v[2]};
    Vec3 va = {v.v[3], v.v[4], v.v[5]};
    Vec3 wxr = cross3(va, r);
    Vec3 tmp = vec3_add(vl, wxr);
    Vec3 lin = mat33_vec3_mul(E, tmp);
    Vec3 ang = mat33_vec3_mul(E, va);
    return {{lin.x, lin.y, lin.z, ang.x, ang.y, ang.z}};
}

// Plücker force transform: child -> parent
__device__ Vec6 transform_force(const Mat33& R, Vec3 r, Vec6 f) {
    Vec3 fl = {f.v[0], f.v[1], f.v[2]};
    Vec3 fa = {f.v[3], f.v[4], f.v[5]};
    Vec3 Rf = mat33_vec3_mul(R, fl);
    Vec3 Rt = mat33_vec3_mul(R, fa);
    Vec3 rxRf = cross3(r, Rf);
    return {{Rf.x, Rf.y, Rf.z, Rt.x+rxRf.x, Rt.y+rxRf.y, Rt.z+rxRf.z}};
}

// v ×* f (force cross)
__device__ Vec6 spatial_cross_force(Vec6 v, Vec6 f) {
    Vec3 vl={v.v[0],v.v[1],v.v[2]}, va={v.v[3],v.v[4],v.v[5]};
    Vec3 fl={f.v[0],f.v[1],f.v[2]}, fa={f.v[3],f.v[4],f.v[5]};
    Vec3 t1=cross3(va,fl), t2=cross3(vl,fl), t3=cross3(va,fa);
    return {{t1.x,t1.y,t1.z, t2.x+t3.x,t2.y+t3.y,t2.z+t3.z}};
}

// v ×_vel u
__device__ Vec6 spatial_cross_vel(Vec6 v, Vec6 u) {
    Vec3 vl={v.v[0],v.v[1],v.v[2]}, va={v.v[3],v.v[4],v.v[5]};
    Vec3 ul={u.v[0],u.v[1],u.v[2]}, ua={u.v[3],u.v[4],u.v[5]};
    Vec3 t1=cross3(va,ul), t2=cross3(vl,ua), t3=cross3(va,ua);
    return {{t1.x+t2.x,t1.y+t2.y,t1.z+t2.z, t3.x,t3.y,t3.z}};
}

// 6x6 @ 6-vec
__device__ Vec6 mat66_mul_vec6(const Mat66& M, Vec6 v) {
    Vec6 r;
    for (int i = 0; i < 6; i++) {
        float s = 0.f;
        for (int k = 0; k < 6; k++) s += M.m[i*6+k] * v.v[k];
        r.v[i] = s;
    }
    return r;
}

// Build 6x6 Plücker matrix X = [[E, -E@skew(r)], [0, E]]
__device__ void build_X6(const Mat33& R, Vec3 r, Mat66& X) {
    Mat33 E = mat33_transpose(R);
    for (int i = 0; i < 36; i++) X.m[i] = 0.f;
    // Top-left & bottom-right: E
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            X.m[i*6+j] = mat33_get(E,i,j);
            X.m[(i+3)*6+(j+3)] = mat33_get(E,i,j);
        }
    // Top-right: -E @ skew(r)
    float K[9] = {0,-r.z,r.y, r.z,0,-r.x, -r.y,r.x,0};
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            float s = 0.f;
            for (int k = 0; k < 3; k++) s += mat33_get(E,i,k) * K[k*3+j];
            X.m[i*6+(j+3)] = -s;
        }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main physics step kernel: FK + passive + PD + contact + ABA + integrate
// ─────────────────────────────────────────────────────────────────────────────

__global__ void physics_step_kernel(
    // Dynamic state (N, nq) / (N, nv)
    float* __restrict__ q,
    float* __restrict__ qdot,
    const float* __restrict__ actions,   // (N, nu)
    // Static robot data
    const int* __restrict__ joint_type,  // (nb,)
    const float* __restrict__ joint_axis, // (nb, 3)
    const int* __restrict__ parent_idx,  // (nb,)
    const int* __restrict__ q_idx_start, // (nb,)
    const int* __restrict__ q_idx_len,   // (nb,)
    const int* __restrict__ v_idx_start, // (nb,)
    const int* __restrict__ v_idx_len,   // (nb,)
    const float* __restrict__ X_tree_R,  // (nb, 3, 3)
    const float* __restrict__ X_tree_r,  // (nb, 3)
    const float* __restrict__ inertia_mat, // (nb, 6, 6)
    const float* __restrict__ q_min,     // (nb,)
    const float* __restrict__ q_max,     // (nb,)
    const float* __restrict__ k_limit,   // (nb,)
    const float* __restrict__ b_limit,   // (nb,)
    const float* __restrict__ damping,   // (nb,)
    const int* __restrict__ actuated_q_idx, // (nu,)
    const int* __restrict__ actuated_v_idx, // (nu,)
    const float* __restrict__ effort_limits, // (nu,)
    const float* __restrict__ contact_body_idx_f, // (nc,) as float for simplicity
    const float* __restrict__ contact_local_pos,  // (nc, 3)
    // Scalar params
    int N_envs, int nb, int nq, int nv, int nu, int nc,
    int has_effort_limits,
    float dt, float gravity,
    float kp, float kd, float action_scale, float action_clip,
    float contact_k_normal, float contact_b_normal, float contact_mu,
    float contact_slip_eps, float contact_ground_z,
    // Outputs
    float* __restrict__ q_new,           // (N, nq)
    float* __restrict__ qdot_new,        // (N, nv)
    float* __restrict__ X_world_R_out,   // (N, nb, 3, 3)
    float* __restrict__ X_world_r_out,   // (N, nb, 3)
    float* __restrict__ v_bodies_out,    // (N, nb, 6)
    int* __restrict__ contact_mask_out   // (N, nc)
) {
    int env_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (env_id >= N_envs) return;

    // Aliases for this env's state
    float* q_e = q + env_id * nq;
    float* qdot_e = qdot + env_id * nv;
    const float* act_e = actions + env_id * nu;

    // ── 1. Passive torques ──
    float tau_passive[64]; // MAX nv
    for (int j = 0; j < nv; j++) tau_passive[j] = 0.f;
    for (int i = 0; i < nb; i++) {
        int jt = joint_type[i];
        if (jt == JOINT_REVOLUTE) {
            int vs = v_idx_start[i], qs = q_idx_start[i];
            float angle = q_e[qs], omega = qdot_e[vs];
            float t = 0.f;
            if (angle < q_min[i]) {
                float pen = q_min[i] - angle;
                t = k_limit[i] * pen - b_limit[i] * fminf(omega, 0.f);
            } else if (angle > q_max[i]) {
                float pen = angle - q_max[i];
                t = -(k_limit[i] * pen + b_limit[i] * fmaxf(omega, 0.f));
            }
            t -= damping[i] * omega;
            tau_passive[vs] = t;
        } else if (jt == JOINT_PRISMATIC) {
            int vs = v_idx_start[i];
            tau_passive[vs] = -damping[i] * qdot_e[vs];
        }
    }

    // ── 2. PD controller ──
    float tau_total[64];
    for (int j = 0; j < nv; j++) tau_total[j] = tau_passive[j];
    for (int j = 0; j < nu; j++) {
        float act = act_e[j];
        if (action_clip > 0.f) act = fmaxf(-action_clip, fminf(action_clip, act));
        int qi = actuated_q_idx[j], vi = actuated_v_idx[j];
        float target = q_e[qi] + act * action_scale;
        float tv = kp * (target - q_e[qi]) - kd * qdot_e[vi];
        if (has_effort_limits) tv = fmaxf(-effort_limits[j], fminf(effort_limits[j], tv));
        tau_total[vi] += tv;
    }

    // ── 3. FK + body velocities ──
    Mat33 X_up_R_arr[MAX_BODIES];
    Vec3  X_up_r_arr[MAX_BODIES];
    Mat33 X_world_R_arr[MAX_BODIES];
    Vec3  X_world_r_arr[MAX_BODIES];
    Vec6  v_bodies[MAX_BODIES];

    for (int i = 0; i < nb; i++) {
        int jt = joint_type[i], qs = q_idx_start[i], vs = v_idx_start[i];
        int vl = v_idx_len[i], pid = parent_idx[i];
        Vec3 axis = {joint_axis[i*3], joint_axis[i*3+1], joint_axis[i*3+2]};

        // Joint transform
        Mat33 R_J = mat33_identity();
        Vec3 r_J = {0,0,0};
        if (jt == JOINT_REVOLUTE) R_J = rodrigues(axis, q_e[qs]);
        else if (jt == JOINT_PRISMATIC) r_J = vec3_scale(axis, q_e[qs]);
        else if (jt == JOINT_FREE) {
            R_J = quat_to_rot(q_e[qs], q_e[qs+1], q_e[qs+2], q_e[qs+3]);
            r_J = {q_e[qs+4], q_e[qs+5], q_e[qs+6]};
        }

        // X_tree
        Mat33 R_tree; Vec3 r_tree;
        for (int a = 0; a < 9; a++) R_tree.m[a] = X_tree_R[i*9+a];
        r_tree = {X_tree_r[i*3], X_tree_r[i*3+1], X_tree_r[i*3+2]};

        // X_up = X_tree @ X_J
        Mat33 R_up = mat33_mul(R_tree, R_J);
        Vec3 r_up = vec3_add(r_tree, mat33_vec3_mul(R_tree, r_J));
        X_up_R_arr[i] = R_up;
        X_up_r_arr[i] = r_up;

        // X_world
        if (pid < 0) {
            X_world_R_arr[i] = R_up;
            X_world_r_arr[i] = r_up;
        } else {
            X_world_R_arr[i] = mat33_mul(X_world_R_arr[pid], R_up);
            X_world_r_arr[i] = vec3_add(X_world_r_arr[pid],
                mat33_vec3_mul(X_world_R_arr[pid], r_up));
        }

        // vJ
        Vec6 vJ = {{0,0,0,0,0,0}};
        if (jt == JOINT_REVOLUTE) {
            float qd = qdot_e[vs];
            vJ.v[3] = axis.x*qd; vJ.v[4] = axis.y*qd; vJ.v[5] = axis.z*qd;
        } else if (jt == JOINT_PRISMATIC) {
            float qd = qdot_e[vs];
            vJ.v[0] = axis.x*qd; vJ.v[1] = axis.y*qd; vJ.v[2] = axis.z*qd;
        } else if (jt == JOINT_FREE) {
            for (int d = 0; d < 6; d++) vJ.v[d] = qdot_e[vs+d];
        }

        if (pid < 0) {
            v_bodies[i] = vJ;
        } else {
            Vec6 vx = transform_velocity(R_up, r_up, v_bodies[pid]);
            for (int d = 0; d < 6; d++) v_bodies[i].v[d] = vx.v[d] + vJ.v[d];
        }
    }

    // ── 4. Contact forces ──
    Vec6 ext_forces[MAX_BODIES];
    for (int i = 0; i < nb; i++) for (int d = 0; d < 6; d++) ext_forces[i].v[d] = 0.f;
    int* cmask = contact_mask_out + env_id * nc;
    for (int c = 0; c < nc; c++) cmask[c] = 0;

    for (int c = 0; c < nc; c++) {
        int bi = __float2int_rn(contact_body_idx_f[c]);
        Vec3 lp = {contact_local_pos[c*3], contact_local_pos[c*3+1], contact_local_pos[c*3+2]};
        Vec3 pw = vec3_add(mat33_vec3_mul(X_world_R_arr[bi], lp), X_world_r_arr[bi]);
        float depth = contact_ground_z - pw.z;
        if (depth > 0.f) {
            cmask[c] = 1;
            Vec3 rlw = mat33_vec3_mul(X_world_R_arr[bi], lp);
            Vec3 vl = mat33_vec3_mul(X_world_R_arr[bi], {v_bodies[bi].v[0],v_bodies[bi].v[1],v_bodies[bi].v[2]});
            Vec3 ow = mat33_vec3_mul(X_world_R_arr[bi], {v_bodies[bi].v[3],v_bodies[bi].v[4],v_bodies[bi].v[5]});
            Vec3 vel = vec3_add(vl, cross3(ow, rlw));

            float Fn = fmaxf(0.f, contact_k_normal * depth - contact_b_normal * vel.z);
            float sn = sqrtf(vel.x*vel.x + vel.y*vel.y + contact_slip_eps*contact_slip_eps);
            float Ftx = -contact_mu * Fn * vel.x / sn;
            float Fty = -contact_mu * Fn * vel.y / sn;
            Vec3 Fw = {Ftx, Fty, Fn};
            Vec3 rarm = vec3_sub(pw, X_world_r_arr[bi]);
            Vec3 tw = cross3(rarm, Fw);
            Vec6 fw = {{Fw.x,Fw.y,Fw.z, tw.x,tw.y,tw.z}};

            Mat33 Rinv = mat33_transpose(X_world_R_arr[bi]);
            Vec3 rinv = vec3_neg(mat33_vec3_mul(Rinv, X_world_r_arr[bi]));
            Vec6 fb = transform_force(Rinv, rinv, fw);
            for (int d = 0; d < 6; d++) ext_forces[bi].v[d] += fb.v[d];
        }
    }

    // ── 5. ABA ──
    Vec6 aba_v[MAX_BODIES], aba_c[MAX_BODIES], aba_pA[MAX_BODIES], aba_a[MAX_BODIES];
    Mat66 aba_IA[MAX_BODIES];
    // Pass 2 storage
    Vec6 aba_U[MAX_BODIES]; float aba_Dinv[MAX_BODIES], aba_u[MAX_BODIES];
    // FreeJoint
    Mat66 aba_IA_free; Vec6 aba_u6;

    float a_grav[6] = {0,0,-gravity,0,0,0};
    Vec6 neg_grav = {{0,0,gravity,0,0,0}};

    // Pass 1
    for (int i = 0; i < nb; i++) {
        int jt = joint_type[i], vs = v_idx_start[i], vl = v_idx_len[i], pid = parent_idx[i];
        Vec3 axis = {joint_axis[i*3], joint_axis[i*3+1], joint_axis[i*3+2]};

        Vec6 vJ = {{0,0,0,0,0,0}};
        if (jt == JOINT_REVOLUTE) {
            float qd = qdot_e[vs]; vJ.v[3]=axis.x*qd; vJ.v[4]=axis.y*qd; vJ.v[5]=axis.z*qd;
        } else if (jt == JOINT_PRISMATIC) {
            float qd = qdot_e[vs]; vJ.v[0]=axis.x*qd; vJ.v[1]=axis.y*qd; vJ.v[2]=axis.z*qd;
        } else if (jt == JOINT_FREE) {
            for (int d = 0; d < 6; d++) vJ.v[d] = qdot_e[vs+d];
        }

        if (pid < 0) {
            aba_v[i] = vJ;
            for (int d = 0; d < 6; d++) aba_c[i].v[d] = 0.f;
        } else {
            Vec6 vx = transform_velocity(X_up_R_arr[i], X_up_r_arr[i], aba_v[pid]);
            for (int d = 0; d < 6; d++) aba_v[i].v[d] = vx.v[d] + vJ.v[d];
            aba_c[i] = spatial_cross_vel(aba_v[i], vJ);
        }

        for (int a = 0; a < 36; a++) aba_IA[i].m[a] = inertia_mat[i*36+a];
        Vec6 Iv = mat66_mul_vec6(aba_IA[i], aba_v[i]);
        Vec6 pA_i = spatial_cross_force(aba_v[i], Iv);
        for (int d = 0; d < 6; d++) aba_pA[i].v[d] = pA_i.v[d] - ext_forces[i].v[d];
    }

    // Pass 2
    for (int idx = 0; idx < nb; idx++) {
        int i = nb - 1 - idx;
        int jt = joint_type[i], vs = v_idx_start[i], vl = v_idx_len[i], pid = parent_idx[i];
        Vec3 axis = {joint_axis[i*3], joint_axis[i*3+1], joint_axis[i*3+2]};

        Mat66 IA_A; Vec6 pA_A;
        for (int a = 0; a < 36; a++) IA_A.m[a] = aba_IA[i].m[a];

        if (vl > 0 && (jt == JOINT_REVOLUTE || jt == JOINT_PRISMATIC)) {
            Vec6 S = {{0,0,0,0,0,0}};
            if (jt == JOINT_REVOLUTE) { S.v[3]=axis.x; S.v[4]=axis.y; S.v[5]=axis.z; }
            else { S.v[0]=axis.x; S.v[1]=axis.y; S.v[2]=axis.z; }

            Vec6 U = mat66_mul_vec6(aba_IA[i], S);
            float D = 0.f;
            for (int d = 0; d < 6; d++) D += S.v[d] * U.v[d];
            float Di = 1.f / D;
            float u = tau_total[vs];
            for (int d = 0; d < 6; d++) u -= S.v[d] * aba_pA[i].v[d];

            aba_U[i] = U; aba_Dinv[i] = Di; aba_u[i] = u;

            for (int a = 0; a < 6; a++)
                for (int b = 0; b < 6; b++)
                    IA_A.m[a*6+b] = aba_IA[i].m[a*6+b] - U.v[a]*U.v[b]*Di;

            Vec6 IAc = mat66_mul_vec6(IA_A, aba_c[i]);
            for (int d = 0; d < 6; d++)
                pA_A.v[d] = aba_pA[i].v[d] + IAc.v[d] + U.v[d]*Di*u;

        } else if (vl > 0 && jt == JOINT_FREE) {
            for (int d = 0; d < 6; d++) aba_u6.v[d] = tau_total[vs+d] - aba_pA[i].v[d];
            for (int a = 0; a < 36; a++) aba_IA_free.m[a] = aba_IA[i].m[a];
            for (int a = 0; a < 36; a++) IA_A.m[a] = 0.f;
            for (int d = 0; d < 6; d++) pA_A.v[d] = aba_pA[i].v[d] + aba_u6.v[d];
        } else {
            Vec6 IAc = mat66_mul_vec6(IA_A, aba_c[i]);
            for (int d = 0; d < 6; d++) pA_A.v[d] = aba_pA[i].v[d] + IAc.v[d];
        }

        if (pid >= 0) {
            Mat66 X6; build_X6(X_up_R_arr[i], X_up_r_arr[i], X6);
            // IA[parent] += X^T @ IA_A @ X
            Mat66 tmp66, contrib;
            for (int a = 0; a < 6; a++)
                for (int b = 0; b < 6; b++) {
                    float s = 0.f;
                    for (int k = 0; k < 6; k++) s += IA_A.m[a*6+k]*X6.m[k*6+b];
                    tmp66.m[a*6+b] = s;
                }
            for (int a = 0; a < 6; a++)
                for (int b = 0; b < 6; b++) {
                    float s = 0.f;
                    for (int k = 0; k < 6; k++) s += X6.m[k*6+a]*tmp66.m[k*6+b];
                    aba_IA[pid].m[a*6+b] += s;
                }
            Vec6 pf = transform_force(X_up_R_arr[i], X_up_r_arr[i], pA_A);
            for (int d = 0; d < 6; d++) aba_pA[pid].v[d] += pf.v[d];
        }
    }

    // Pass 3
    float qddot[64];
    for (int j = 0; j < nv; j++) qddot[j] = 0.f;

    for (int i = 0; i < nb; i++) {
        int jt = joint_type[i], vs = v_idx_start[i], vl = v_idx_len[i], pid = parent_idx[i];
        Vec3 axis = {joint_axis[i*3], joint_axis[i*3+1], joint_axis[i*3+2]};

        Vec6 a_p;
        if (pid < 0) a_p = transform_velocity(X_up_R_arr[i], X_up_r_arr[i], neg_grav);
        else a_p = transform_velocity(X_up_R_arr[i], X_up_r_arr[i], aba_a[pid]);

        Vec6 apc;
        for (int d = 0; d < 6; d++) apc.v[d] = a_p.v[d] + aba_c[i].v[d];

        if (vl > 0 && (jt == JOINT_REVOLUTE || jt == JOINT_PRISMATIC)) {
            float UT_apc = 0.f;
            for (int d = 0; d < 6; d++) UT_apc += aba_U[i].v[d] * apc.v[d];
            float qdd_i = aba_Dinv[i] * (aba_u[i] - UT_apc);
            qddot[vs] = qdd_i;

            Vec6 S = {{0,0,0,0,0,0}};
            if (jt == JOINT_REVOLUTE) { S.v[3]=axis.x; S.v[4]=axis.y; S.v[5]=axis.z; }
            else { S.v[0]=axis.x; S.v[1]=axis.y; S.v[2]=axis.z; }
            for (int d = 0; d < 6; d++) aba_a[i].v[d] = apc.v[d] + S.v[d]*qdd_i;

        } else if (vl > 0 && jt == JOINT_FREE) {
            Vec6 IA_apc = mat66_mul_vec6(aba_IA_free, apc);
            float rhs[6];
            for (int d = 0; d < 6; d++) rhs[d] = aba_u6.v[d] - IA_apc.v[d];
            // Gaussian elimination
            float aug[6][7];
            for (int a = 0; a < 6; a++) {
                for (int b = 0; b < 6; b++) aug[a][b] = aba_IA_free.m[a*6+b];
                aug[a][6] = rhs[a];
            }
            for (int col = 0; col < 6; col++) {
                for (int row = col+1; row < 6; row++) {
                    float f = aug[row][col] / aug[col][col];
                    for (int k = col; k < 7; k++) aug[row][k] -= f * aug[col][k];
                }
            }
            float qdd[6];
            for (int row = 5; row >= 0; row--) {
                float s = aug[row][6];
                for (int k = row+1; k < 6; k++) s -= aug[row][k] * qdd[k];
                qdd[row] = s / aug[row][row];
            }
            for (int d = 0; d < 6; d++) {
                qddot[vs+d] = qdd[d];
                aba_a[i].v[d] = apc.v[d] + qdd[d];
            }
        } else {
            for (int d = 0; d < 6; d++) aba_a[i].v[d] = apc.v[d];
        }
    }

    // ── 6. Integration (semi-implicit Euler) ──
    float* qn = q_new + env_id * nq;
    float* vn = qdot_new + env_id * nv;

    for (int j = 0; j < nv; j++) vn[j] = qdot_e[j] + dt * qddot[j];

    for (int i = 0; i < nb; i++) {
        int jt = joint_type[i], qs = q_idx_start[i], ql = q_idx_len[i];
        int vs = v_idx_start[i], vl = v_idx_len[i];

        if (jt == JOINT_FREE) {
            float qw=q_e[qs], qx=q_e[qs+1], qy=q_e[qs+2], qz=q_e[qs+3];
            float wx=vn[vs+3], wy=vn[vs+4], wz=vn[vs+5];
            float dqw = .5f*(-qx*wx - qy*wy - qz*wz);
            float dqx = .5f*(qw*wx + qy*wz - qz*wy);
            float dqy = .5f*(qw*wy - qx*wz + qz*wx);
            float dqz = .5f*(qw*wz + qx*wy - qy*wx);
            float nqw=qw+dqw*dt, nqx=qx+dqx*dt, nqy=qy+dqy*dt, nqz=qz+dqz*dt;
            float n = sqrtf(nqw*nqw+nqx*nqx+nqy*nqy+nqz*nqz);
            qn[qs]=nqw/n; qn[qs+1]=nqx/n; qn[qs+2]=nqy/n; qn[qs+3]=nqz/n;
            qn[qs+4]=q_e[qs+4]+dt*vn[vs]; qn[qs+5]=q_e[qs+5]+dt*vn[vs+1]; qn[qs+6]=q_e[qs+6]+dt*vn[vs+2];
        } else if (vl > 0) {
            for (int k = 0; k < ql; k++) qn[qs+k] = q_e[qs+k] + dt*vn[vs+k];
        }
    }

    // ── 7. Write FK outputs (re-run FK on new state would be ideal, but we output current) ──
    for (int i = 0; i < nb; i++) {
        float* Rout = X_world_R_out + (env_id*nb+i)*9;
        float* rout = X_world_r_out + (env_id*nb+i)*3;
        float* vout = v_bodies_out + (env_id*nb+i)*6;
        for (int a = 0; a < 9; a++) Rout[a] = X_world_R_arr[i].m[a];
        rout[0]=X_world_r_arr[i].x; rout[1]=X_world_r_arr[i].y; rout[2]=X_world_r_arr[i].z;
        for (int d = 0; d < 6; d++) vout[d] = v_bodies[i].v[d];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FK-only kernel (for reset / observation cache)
// ─────────────────────────────────────────────────────────────────────────────

__global__ void fk_only_kernel(
    const float* __restrict__ q,
    const float* __restrict__ qdot,
    const int* __restrict__ joint_type,
    const float* __restrict__ joint_axis,
    const int* __restrict__ parent_idx,
    const int* __restrict__ q_idx_start,
    const int* __restrict__ v_idx_start,
    const int* __restrict__ v_idx_len,
    const float* __restrict__ X_tree_R,
    const float* __restrict__ X_tree_r,
    int N, int nb, int nq, int nv,
    float* __restrict__ X_world_R_out,
    float* __restrict__ X_world_r_out,
    float* __restrict__ v_bodies_out
) {
    int env_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (env_id >= N) return;
    const float* q_e = q + env_id * nq;
    const float* qdot_e = qdot + env_id * nv;

    Mat33 Xw_R[MAX_BODIES]; Vec3 Xw_r[MAX_BODIES]; Vec6 vb[MAX_BODIES];

    for (int i = 0; i < nb; i++) {
        int jt = joint_type[i], qs = q_idx_start[i], vs = v_idx_start[i];
        int vl = v_idx_len[i], pid = parent_idx[i];
        Vec3 axis = {joint_axis[i*3], joint_axis[i*3+1], joint_axis[i*3+2]};

        Mat33 R_J = mat33_identity(); Vec3 r_J = {0,0,0};
        if (jt == JOINT_REVOLUTE) R_J = rodrigues(axis, q_e[qs]);
        else if (jt == JOINT_PRISMATIC) r_J = vec3_scale(axis, q_e[qs]);
        else if (jt == JOINT_FREE) {
            R_J = quat_to_rot(q_e[qs], q_e[qs+1], q_e[qs+2], q_e[qs+3]);
            r_J = {q_e[qs+4], q_e[qs+5], q_e[qs+6]};
        }

        Mat33 R_tree; for (int a=0;a<9;a++) R_tree.m[a]=X_tree_R[i*9+a];
        Vec3 r_tree = {X_tree_r[i*3], X_tree_r[i*3+1], X_tree_r[i*3+2]};
        Mat33 R_up = mat33_mul(R_tree, R_J);
        Vec3 r_up = vec3_add(r_tree, mat33_vec3_mul(R_tree, r_J));

        if (pid<0) { Xw_R[i]=R_up; Xw_r[i]=r_up; }
        else { Xw_R[i]=mat33_mul(Xw_R[pid],R_up); Xw_r[i]=vec3_add(Xw_r[pid],mat33_vec3_mul(Xw_R[pid],r_up)); }

        Vec6 vJ = {{0,0,0,0,0,0}};
        if (jt==JOINT_REVOLUTE) { float qd=qdot_e[vs]; vJ.v[3]=axis.x*qd; vJ.v[4]=axis.y*qd; vJ.v[5]=axis.z*qd; }
        else if (jt==JOINT_PRISMATIC) { float qd=qdot_e[vs]; vJ.v[0]=axis.x*qd; vJ.v[1]=axis.y*qd; vJ.v[2]=axis.z*qd; }
        else if (jt==JOINT_FREE) { for(int d=0;d<6;d++) vJ.v[d]=qdot_e[vs+d]; }

        if (pid<0) vb[i]=vJ;
        else {
            Vec6 vx=transform_velocity(R_up,r_up,vb[pid]);
            for(int d=0;d<6;d++) vb[i].v[d]=vx.v[d]+vJ.v[d];
        }
    }

    for (int i=0;i<nb;i++) {
        float* Ro=X_world_R_out+(env_id*nb+i)*9;
        float* ro=X_world_r_out+(env_id*nb+i)*3;
        float* vo=v_bodies_out+(env_id*nb+i)*6;
        for(int a=0;a<9;a++) Ro[a]=Xw_R[i].m[a];
        ro[0]=Xw_r[i].x; ro[1]=Xw_r[i].y; ro[2]=Xw_r[i].z;
        for(int d=0;d<6;d++) vo[d]=vb[i].v[d];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PyTorch binding
// ─────────────────────────────────────────────────────────────────────────────

std::vector<torch::Tensor> physics_step(
    torch::Tensor q, torch::Tensor qdot, torch::Tensor actions,
    torch::Tensor joint_type, torch::Tensor joint_axis,
    torch::Tensor parent_idx, torch::Tensor q_idx_start, torch::Tensor q_idx_len,
    torch::Tensor v_idx_start, torch::Tensor v_idx_len,
    torch::Tensor X_tree_R, torch::Tensor X_tree_r,
    torch::Tensor inertia_mat,
    torch::Tensor q_min_t, torch::Tensor q_max_t,
    torch::Tensor k_limit_t, torch::Tensor b_limit_t, torch::Tensor damping_t,
    torch::Tensor actuated_q_idx, torch::Tensor actuated_v_idx,
    torch::Tensor effort_limits_t,
    torch::Tensor contact_body_idx, torch::Tensor contact_local_pos,
    int N, int nb, int nq, int nv, int nu, int nc,
    int has_effort_limits,
    float dt, float gravity,
    float kp, float kd, float action_scale, float action_clip,
    float contact_k_normal, float contact_b_normal, float contact_mu,
    float contact_slip_eps, float contact_ground_z
) {
    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    auto opts_i = torch::TensorOptions().dtype(torch::kInt32).device(q.device());

    auto q_new = torch::zeros({N, nq}, opts_f);
    auto qdot_new = torch::zeros({N, nv}, opts_f);
    auto X_world_R_out = torch::zeros({N, nb, 3, 3}, opts_f);
    auto X_world_r_out = torch::zeros({N, nb, 3}, opts_f);
    auto v_bodies_out = torch::zeros({N, nb, 6}, opts_f);
    auto contact_mask_out = torch::zeros({N, nc}, opts_i);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    physics_step_kernel<<<blocks, threads>>>(
        q.data_ptr<float>(), qdot.data_ptr<float>(), actions.data_ptr<float>(),
        joint_type.data_ptr<int>(), joint_axis.data_ptr<float>(),
        parent_idx.data_ptr<int>(), q_idx_start.data_ptr<int>(), q_idx_len.data_ptr<int>(),
        v_idx_start.data_ptr<int>(), v_idx_len.data_ptr<int>(),
        X_tree_R.data_ptr<float>(), X_tree_r.data_ptr<float>(),
        inertia_mat.data_ptr<float>(),
        q_min_t.data_ptr<float>(), q_max_t.data_ptr<float>(),
        k_limit_t.data_ptr<float>(), b_limit_t.data_ptr<float>(), damping_t.data_ptr<float>(),
        actuated_q_idx.data_ptr<int>(), actuated_v_idx.data_ptr<int>(),
        effort_limits_t.data_ptr<float>(),
        contact_body_idx.data_ptr<float>(), contact_local_pos.data_ptr<float>(),
        N, nb, nq, nv, nu, nc, has_effort_limits,
        dt, gravity, kp, kd, action_scale, action_clip,
        contact_k_normal, contact_b_normal, contact_mu, contact_slip_eps, contact_ground_z,
        q_new.data_ptr<float>(), qdot_new.data_ptr<float>(),
        X_world_R_out.data_ptr<float>(), X_world_r_out.data_ptr<float>(),
        v_bodies_out.data_ptr<float>(), contact_mask_out.data_ptr<int>()
    );
    return {q_new, qdot_new, X_world_R_out, X_world_r_out, v_bodies_out, contact_mask_out};
}

std::vector<torch::Tensor> fk_only(
    torch::Tensor q, torch::Tensor qdot,
    torch::Tensor joint_type, torch::Tensor joint_axis,
    torch::Tensor parent_idx, torch::Tensor q_idx_start,
    torch::Tensor v_idx_start, torch::Tensor v_idx_len,
    torch::Tensor X_tree_R, torch::Tensor X_tree_r,
    int N, int nb, int nq, int nv
) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    auto Rout = torch::zeros({N, nb, 3, 3}, opts);
    auto rout = torch::zeros({N, nb, 3}, opts);
    auto vout = torch::zeros({N, nb, 6}, opts);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    fk_only_kernel<<<blocks, threads>>>(
        q.data_ptr<float>(), qdot.data_ptr<float>(),
        joint_type.data_ptr<int>(), joint_axis.data_ptr<float>(),
        parent_idx.data_ptr<int>(), q_idx_start.data_ptr<int>(),
        v_idx_start.data_ptr<int>(), v_idx_len.data_ptr<int>(),
        X_tree_R.data_ptr<float>(), X_tree_r.data_ptr<float>(),
        N, nb, nq, nv,
        Rout.data_ptr<float>(), rout.data_ptr<float>(), vout.data_ptr<float>()
    );
    return {Rout, rout, vout};
}

// ─────────────────────────────────────────────────────────────────────────────
// Fused CRBA kernel: FK + CRBA(H) + RNEA(C) + Cholesky solve, single launch
// ─────────────────────────────────────────────────────────────────────────────

#define MAX_NV 48

// 6x6 matrix multiply into flat array region
__device__ void mat66_mul_to(const float* A, const float* B, float* C) {
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++) {
            float s = 0.f;
            for (int k = 0; k < 6; k++) s += A[i*6+k] * B[k*6+j];
            C[i*6+j] = s;
        }
}

// X^T @ M @ X  (all 6x6, result added to dst)
__device__ void XtMX_add(const Mat33& R_up, Vec3 r_up, const float* M, float* dst) {
    // Build X6 (6x6)
    Mat66 X6; build_X6(R_up, r_up, X6);
    // tmp = M @ X
    float tmp[36];
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++) {
            float s = 0.f;
            for (int k = 0; k < 6; k++) s += M[i*6+k] * X6.m[k*6+j];
            tmp[i*6+j] = s;
        }
    // dst += X^T @ tmp
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++) {
            float s = 0.f;
            for (int k = 0; k < 6; k++) s += X6.m[k*6+i] * tmp[k*6+j];
            dst[i*6+j] += s;
        }
}

// Apply force transform to multiple columns: F (6 x ncols) -> F' (6 x ncols)
__device__ void apply_force_cols(const Mat33& R, Vec3 r, float* F, int ncols) {
    for (int c = 0; c < ncols; c++) {
        float fl[3] = {F[0*ncols+c], F[1*ncols+c], F[2*ncols+c]};
        float fa[3] = {F[3*ncols+c], F[4*ncols+c], F[5*ncols+c]};
        float Rf[3], Rt[3];
        for (int i = 0; i < 3; i++) {
            Rf[i] = R.m[i*3]*fl[0] + R.m[i*3+1]*fl[1] + R.m[i*3+2]*fl[2];
            Rt[i] = R.m[i*3]*fa[0] + R.m[i*3+1]*fa[1] + R.m[i*3+2]*fa[2];
        }
        float rxRf[3] = {r.y*Rf[2]-r.z*Rf[1], r.z*Rf[0]-r.x*Rf[2], r.x*Rf[1]-r.y*Rf[0]};
        F[0*ncols+c]=Rf[0]; F[1*ncols+c]=Rf[1]; F[2*ncols+c]=Rf[2];
        F[3*ncols+c]=Rt[0]+rxRf[0]; F[4*ncols+c]=Rt[1]+rxRf[1]; F[5*ncols+c]=Rt[2]+rxRf[2];
    }
}

__global__ void physics_step_crba_kernel(
    float* __restrict__ q,
    float* __restrict__ qdot,
    const float* __restrict__ actions,
    const int* __restrict__ joint_type,
    const float* __restrict__ joint_axis,
    const int* __restrict__ parent_idx,
    const int* __restrict__ q_idx_start,
    const int* __restrict__ q_idx_len,
    const int* __restrict__ v_idx_start,
    const int* __restrict__ v_idx_len,
    const float* __restrict__ X_tree_R,
    const float* __restrict__ X_tree_r,
    const float* __restrict__ inertia_mat,
    const float* __restrict__ q_min,
    const float* __restrict__ q_max,
    const float* __restrict__ k_limit,
    const float* __restrict__ b_limit,
    const float* __restrict__ damping,
    const int* __restrict__ actuated_q_idx,
    const int* __restrict__ actuated_v_idx,
    const float* __restrict__ effort_limits,
    const float* __restrict__ contact_body_idx_f,
    const float* __restrict__ contact_local_pos,
    int N_envs, int nb, int nq, int nv, int nu, int nc,
    int has_effort_limits,
    float dt, float gravity,
    float kp, float kd, float action_scale, float action_clip,
    float contact_k_normal, float contact_b_normal, float contact_mu,
    float contact_slip_eps, float contact_ground_z,
    float* __restrict__ q_new,
    float* __restrict__ qdot_new,
    float* __restrict__ X_world_R_out,
    float* __restrict__ X_world_r_out,
    float* __restrict__ v_bodies_out,
    int* __restrict__ contact_mask_out
) {
    int env_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (env_id >= N_envs) return;

    float* q_e = q + env_id * nq;
    float* qdot_e = qdot + env_id * nv;
    const float* act_e = actions + env_id * nu;

    // ── 1. Passive torques + PD controller → tau_total ──
    float tau_total[MAX_NV];
    for (int j = 0; j < nv; j++) tau_total[j] = 0.f;
    for (int i = 0; i < nb; i++) {
        int jt = joint_type[i];
        if (jt == JOINT_REVOLUTE) {
            int vs = v_idx_start[i], qs = q_idx_start[i];
            float angle = q_e[qs], omega = qdot_e[vs];
            float t = 0.f;
            if (angle < q_min[i]) t = k_limit[i]*(q_min[i]-angle) - b_limit[i]*fminf(omega,0.f);
            else if (angle > q_max[i]) t = -(k_limit[i]*(angle-q_max[i]) + b_limit[i]*fmaxf(omega,0.f));
            tau_total[vs] = t - damping[i]*omega;
        } else if (jt == JOINT_PRISMATIC) {
            tau_total[v_idx_start[i]] = -damping[i]*qdot_e[v_idx_start[i]];
        }
    }
    for (int j = 0; j < nu; j++) {
        float act = act_e[j];
        if (action_clip > 0.f) act = fmaxf(-action_clip, fminf(action_clip, act));
        int qi = actuated_q_idx[j], vi = actuated_v_idx[j];
        float tv = kp*(q_e[qi] + act*action_scale - q_e[qi]) - kd*qdot_e[vi];
        if (has_effort_limits) tv = fmaxf(-effort_limits[j], fminf(effort_limits[j], tv));
        tau_total[vi] += tv;
    }

    // ── 2. FK (same as ABA) ──
    Mat33 X_up_R_arr[MAX_BODIES]; Vec3 X_up_r_arr[MAX_BODIES];
    Mat33 X_world_R_arr[MAX_BODIES]; Vec3 X_world_r_arr[MAX_BODIES];
    Vec6 v_bodies[MAX_BODIES];

    for (int i = 0; i < nb; i++) {
        int jt = joint_type[i], qs = q_idx_start[i], vs = v_idx_start[i];
        int vl = v_idx_len[i], pid = parent_idx[i];
        Vec3 axis = {joint_axis[i*3], joint_axis[i*3+1], joint_axis[i*3+2]};
        Mat33 R_J = mat33_identity(); Vec3 r_J = {0,0,0};
        if (jt == JOINT_REVOLUTE) R_J = rodrigues(axis, q_e[qs]);
        else if (jt == JOINT_PRISMATIC) r_J = vec3_scale(axis, q_e[qs]);
        else if (jt == JOINT_FREE) {
            R_J = quat_to_rot(q_e[qs], q_e[qs+1], q_e[qs+2], q_e[qs+3]);
            r_J = {q_e[qs+4], q_e[qs+5], q_e[qs+6]};
        }
        Mat33 R_tree; for (int a=0;a<9;a++) R_tree.m[a]=X_tree_R[i*9+a];
        Vec3 r_tree = {X_tree_r[i*3], X_tree_r[i*3+1], X_tree_r[i*3+2]};
        X_up_R_arr[i] = mat33_mul(R_tree, R_J);
        X_up_r_arr[i] = vec3_add(r_tree, mat33_vec3_mul(R_tree, r_J));
        if (pid<0) { X_world_R_arr[i]=X_up_R_arr[i]; X_world_r_arr[i]=X_up_r_arr[i]; }
        else { X_world_R_arr[i]=mat33_mul(X_world_R_arr[pid],X_up_R_arr[i]); X_world_r_arr[i]=vec3_add(X_world_r_arr[pid],mat33_vec3_mul(X_world_R_arr[pid],X_up_r_arr[i])); }
        Vec6 vJ={{0,0,0,0,0,0}};
        if (jt==JOINT_REVOLUTE) { float qd=qdot_e[vs]; vJ.v[3]=axis.x*qd; vJ.v[4]=axis.y*qd; vJ.v[5]=axis.z*qd; }
        else if (jt==JOINT_PRISMATIC) { float qd=qdot_e[vs]; vJ.v[0]=axis.x*qd; vJ.v[1]=axis.y*qd; vJ.v[2]=axis.z*qd; }
        else if (jt==JOINT_FREE) { for(int d=0;d<6;d++) vJ.v[d]=qdot_e[vs+d]; }
        if (pid<0) v_bodies[i]=vJ;
        else { Vec6 vx=transform_velocity(X_up_R_arr[i],X_up_r_arr[i],v_bodies[pid]); for(int d=0;d<6;d++) v_bodies[i].v[d]=vx.v[d]+vJ.v[d]; }
    }

    // ── 3. Contact forces ──
    Vec6 ext_forces[MAX_BODIES];
    for (int i=0;i<nb;i++) for(int d=0;d<6;d++) ext_forces[i].v[d]=0.f;
    int* cmask = contact_mask_out + env_id*nc;
    for (int c=0;c<nc;c++) cmask[c]=0;
    for (int c=0;c<nc;c++) {
        int bi=__float2int_rn(contact_body_idx_f[c]);
        Vec3 lp={contact_local_pos[c*3],contact_local_pos[c*3+1],contact_local_pos[c*3+2]};
        Vec3 pw=vec3_add(mat33_vec3_mul(X_world_R_arr[bi],lp),X_world_r_arr[bi]);
        float depth=contact_ground_z-pw.z;
        if(depth>0.f){
            cmask[c]=1;
            Vec3 rlw=mat33_vec3_mul(X_world_R_arr[bi],lp);
            Vec3 vl=mat33_vec3_mul(X_world_R_arr[bi],{v_bodies[bi].v[0],v_bodies[bi].v[1],v_bodies[bi].v[2]});
            Vec3 ow=mat33_vec3_mul(X_world_R_arr[bi],{v_bodies[bi].v[3],v_bodies[bi].v[4],v_bodies[bi].v[5]});
            Vec3 vel=vec3_add(vl,cross3(ow,rlw));
            float Fn=fmaxf(0.f,contact_k_normal*depth-contact_b_normal*vel.z);
            float sn=sqrtf(vel.x*vel.x+vel.y*vel.y+contact_slip_eps*contact_slip_eps);
            Vec3 Fw={-contact_mu*Fn*vel.x/sn,-contact_mu*Fn*vel.y/sn,Fn};
            Vec3 tw=cross3(vec3_sub(pw,X_world_r_arr[bi]),Fw);
            Vec6 fw={{Fw.x,Fw.y,Fw.z,tw.x,tw.y,tw.z}};
            Mat33 Ri=mat33_transpose(X_world_R_arr[bi]);
            Vec3 ri=vec3_neg(mat33_vec3_mul(Ri,X_world_r_arr[bi]));
            Vec6 fb=transform_force(Ri,ri,fw);
            for(int d=0;d<6;d++) ext_forces[bi].v[d]+=fb.v[d];
        }
    }

    // ── 4. CRBA: composite inertias ──
    float IC[MAX_BODIES * 36];  // nb × 6×6
    for (int i = 0; i < nb; i++)
        for (int a = 0; a < 36; a++)
            IC[i*36+a] = inertia_mat[i*36+a];

    for (int idx = 0; idx < nb; idx++) {
        int i = nb-1-idx;
        int pid = parent_idx[i];
        if (pid >= 0)
            XtMX_add(X_up_R_arr[i], X_up_r_arr[i], &IC[i*36], &IC[pid*36]);
    }

    // ── 5. Build H (nv × nv) ──
    float H[MAX_NV * MAX_NV];
    for (int a = 0; a < nv*nv; a++) H[a] = 0.f;

    for (int i = 0; i < nb; i++) {
        int jt = joint_type[i], vs = v_idx_start[i], vl = v_idx_len[i];
        if (vl == 0) continue;
        Vec3 axis = {joint_axis[i*3], joint_axis[i*3+1], joint_axis[i*3+2]};

        // S_i columns (6 × vl), stored column-major in F (6 × vl)
        float F[6 * 6]; // max vl=6 for free joint
        // F = IC[i] @ S_i
        if (jt == JOINT_REVOLUTE || jt == JOINT_PRISMATIC) {
            float S[6] = {0,0,0,0,0,0};
            if (jt == JOINT_REVOLUTE) { S[3]=axis.x; S[4]=axis.y; S[5]=axis.z; }
            else { S[0]=axis.x; S[1]=axis.y; S[2]=axis.z; }
            // F = IC[i] @ S  (single column)
            for (int r = 0; r < 6; r++) {
                float s = 0.f;
                for (int k = 0; k < 6; k++) s += IC[i*36+r*6+k] * S[k];
                F[r] = s;  // column-major: F[r*1+0]
            }
            // H[vs,vs] = S^T @ F
            float diag = 0.f;
            for (int d = 0; d < 6; d++) diag += S[d] * F[d];
            H[vs*nv+vs] = diag;
        } else if (jt == JOINT_FREE) {
            // S = I6, F = IC[i] (all 6 columns)
            for (int r = 0; r < 6; r++)
                for (int c = 0; c < 6; c++)
                    F[r*6+c] = IC[i*36+r*6+c];  // column-major storage
            // H[vs:vs+6, vs:vs+6] = S^T @ F = F (since S=I)
            for (int a = 0; a < 6; a++)
                for (int b = 0; b < 6; b++)
                    H[(vs+a)*nv+(vs+b)] = F[a*6+b];
        }

        // Propagate F up the tree for off-diagonal blocks
        int j = i;
        while (parent_idx[j] >= 0) {
            apply_force_cols(X_up_R_arr[j], X_up_r_arr[j], F, vl);
            j = parent_idx[j];
            int jt_j = joint_type[j], vs_j = v_idx_start[j], vl_j = v_idx_len[j];
            if (vl_j > 0) {
                Vec3 ax_j = {joint_axis[j*3], joint_axis[j*3+1], joint_axis[j*3+2]};
                if (jt_j == JOINT_REVOLUTE || jt_j == JOINT_PRISMATIC) {
                    float S_j[6] = {0,0,0,0,0,0};
                    if (jt_j == JOINT_REVOLUTE) { S_j[3]=ax_j.x; S_j[4]=ax_j.y; S_j[5]=ax_j.z; }
                    else { S_j[0]=ax_j.x; S_j[1]=ax_j.y; S_j[2]=ax_j.z; }
                    // block = S_j^T @ F  (1 × vl)
                    for (int c = 0; c < vl; c++) {
                        float s = 0.f;
                        for (int d = 0; d < 6; d++) s += S_j[d] * F[d*vl+c];
                        H[vs_j*nv+(vs+c)] = s;
                        H[(vs+c)*nv+vs_j] = s;  // symmetry
                    }
                } else if (jt_j == JOINT_FREE) {
                    // block = I^T @ F = F  (6 × vl)
                    for (int a = 0; a < 6; a++)
                        for (int c = 0; c < vl; c++) {
                            float val = F[a*vl+c];
                            H[(vs_j+a)*nv+(vs+c)] = val;
                            H[(vs+c)*nv+(vs_j+a)] = val;
                        }
                }
            }
        }
    }

    // ── 6. RNEA bias forces (C = RNEA(q, qdot, 0, ext)) ──
    Vec6 rnea_v[MAX_BODIES], rnea_a[MAX_BODIES], rnea_f[MAX_BODIES];
    Vec6 neg_grav = {{0,0,gravity,0,0,0}};

    for (int i = 0; i < nb; i++) {
        int jt=joint_type[i], vs=v_idx_start[i], vl=v_idx_len[i], pid=parent_idx[i];
        Vec3 axis={joint_axis[i*3],joint_axis[i*3+1],joint_axis[i*3+2]};
        Vec6 vJ={{0,0,0,0,0,0}};
        if(jt==JOINT_REVOLUTE){float qd=qdot_e[vs];vJ.v[3]=axis.x*qd;vJ.v[4]=axis.y*qd;vJ.v[5]=axis.z*qd;}
        else if(jt==JOINT_PRISMATIC){float qd=qdot_e[vs];vJ.v[0]=axis.x*qd;vJ.v[1]=axis.y*qd;vJ.v[2]=axis.z*qd;}
        else if(jt==JOINT_FREE){for(int d=0;d<6;d++)vJ.v[d]=qdot_e[vs+d];}

        if(pid<0){
            rnea_v[i]=vJ;
            rnea_a[i]=transform_velocity(X_up_R_arr[i],X_up_r_arr[i],neg_grav);
        } else {
            Vec6 vx=transform_velocity(X_up_R_arr[i],X_up_r_arr[i],rnea_v[pid]);
            for(int d=0;d<6;d++) rnea_v[i].v[d]=vx.v[d]+vJ.v[d];
            Vec6 ax=transform_velocity(X_up_R_arr[i],X_up_r_arr[i],rnea_a[pid]);
            Vec6 cvj=spatial_cross_vel(rnea_v[i],vJ);
            for(int d=0;d<6;d++) rnea_a[i].v[d]=ax.v[d]+cvj.v[d];
        }
        Vec6 Iv=mat66_mul_vec6(*(Mat66*)&inertia_mat[i*36], rnea_v[i]);
        Vec6 Ia=mat66_mul_vec6(*(Mat66*)&inertia_mat[i*36], rnea_a[i]);
        Vec6 vxIv=spatial_cross_force(rnea_v[i], Iv);
        for(int d=0;d<6;d++) rnea_f[i].v[d]=Ia.v[d]+vxIv.v[d]-ext_forces[i].v[d];
    }

    float C[MAX_NV];
    for(int j=0;j<nv;j++) C[j]=0.f;
    for(int idx=0;idx<nb;idx++){
        int i=nb-1-idx;
        int jt=joint_type[i],vs=v_idx_start[i],vl=v_idx_len[i],pid=parent_idx[i];
        Vec3 axis={joint_axis[i*3],joint_axis[i*3+1],joint_axis[i*3+2]};
        if(vl>0){
            if(jt==JOINT_REVOLUTE||jt==JOINT_PRISMATIC){
                float S[6]={0,0,0,0,0,0};
                if(jt==JOINT_REVOLUTE){S[3]=axis.x;S[4]=axis.y;S[5]=axis.z;}
                else{S[0]=axis.x;S[1]=axis.y;S[2]=axis.z;}
                float t=0.f; for(int d=0;d<6;d++) t+=S[d]*rnea_f[i].v[d];
                C[vs]=t;
            } else if(jt==JOINT_FREE){
                for(int d=0;d<6;d++) C[vs+d]=rnea_f[i].v[d];
            }
        }
        if(pid>=0){
            Vec6 fp=transform_force(X_up_R_arr[i],X_up_r_arr[i],rnea_f[i]);
            for(int d=0;d<6;d++) rnea_f[pid].v[d]+=fp.v[d];
        }
    }

    // ── 7. rhs = tau - C ──
    float rhs[MAX_NV];
    for (int j = 0; j < nv; j++) rhs[j] = tau_total[j] - C[j];

    // ── 8. Cholesky factorization L @ L^T = H, then solve ──
    // In-place: H -> L (lower triangular)
    for (int j = 0; j < nv; j++) {
        for (int k = 0; k < j; k++) {
            float s = H[j*nv+k];
            for (int m = 0; m < k; m++) s -= H[j*nv+m] * H[k*nv+m];
            H[j*nv+k] = s / H[k*nv+k];
        }
        float s = H[j*nv+j];
        for (int k = 0; k < j; k++) s -= H[j*nv+k] * H[j*nv+k];
        H[j*nv+j] = sqrtf(s);
    }
    // Forward solve: L @ y = rhs
    float y[MAX_NV];
    for (int i = 0; i < nv; i++) {
        float s = rhs[i];
        for (int k = 0; k < i; k++) s -= H[i*nv+k] * y[k];
        y[i] = s / H[i*nv+i];
    }
    // Back solve: L^T @ qddot = y
    float qddot[MAX_NV];
    for (int i = nv-1; i >= 0; i--) {
        float s = y[i];
        for (int k = i+1; k < nv; k++) s -= H[k*nv+i] * qddot[k];
        qddot[i] = s / H[i*nv+i];
    }

    // ── 9. Integration ──
    float* qn = q_new + env_id*nq;
    float* vn = qdot_new + env_id*nv;
    for(int j=0;j<nv;j++) vn[j]=qdot_e[j]+dt*qddot[j];
    for(int i=0;i<nb;i++){
        int jt=joint_type[i],qs=q_idx_start[i],ql=q_idx_len[i],vs=v_idx_start[i],vl=v_idx_len[i];
        if(jt==JOINT_FREE){
            float qw=q_e[qs],qx=q_e[qs+1],qy=q_e[qs+2],qz=q_e[qs+3];
            float wx=vn[vs+3],wy=vn[vs+4],wz=vn[vs+5];
            float dqw=.5f*(-qx*wx-qy*wy-qz*wz), dqx=.5f*(qw*wx+qy*wz-qz*wy);
            float dqy=.5f*(qw*wy-qx*wz+qz*wx), dqz=.5f*(qw*wz+qx*wy-qy*wx);
            float nqw=qw+dqw*dt,nqx=qx+dqx*dt,nqy=qy+dqy*dt,nqz=qz+dqz*dt;
            float n=sqrtf(nqw*nqw+nqx*nqx+nqy*nqy+nqz*nqz);
            qn[qs]=nqw/n;qn[qs+1]=nqx/n;qn[qs+2]=nqy/n;qn[qs+3]=nqz/n;
            qn[qs+4]=q_e[qs+4]+dt*vn[vs];qn[qs+5]=q_e[qs+5]+dt*vn[vs+1];qn[qs+6]=q_e[qs+6]+dt*vn[vs+2];
        } else if(vl>0){
            for(int k=0;k<ql;k++) qn[qs+k]=q_e[qs+k]+dt*vn[vs+k];
        }
    }
    // Write FK outputs
    for(int i=0;i<nb;i++){
        float*Ro=X_world_R_out+(env_id*nb+i)*9; float*ro=X_world_r_out+(env_id*nb+i)*3; float*vo=v_bodies_out+(env_id*nb+i)*6;
        for(int a=0;a<9;a++) Ro[a]=X_world_R_arr[i].m[a];
        ro[0]=X_world_r_arr[i].x;ro[1]=X_world_r_arr[i].y;ro[2]=X_world_r_arr[i].z;
        for(int d=0;d<6;d++) vo[d]=v_bodies[i].v[d];
    }
}

// C++ wrapper
std::vector<torch::Tensor> physics_step_crba(
    torch::Tensor q, torch::Tensor qdot, torch::Tensor actions,
    torch::Tensor joint_type, torch::Tensor joint_axis,
    torch::Tensor parent_idx, torch::Tensor q_idx_start, torch::Tensor q_idx_len,
    torch::Tensor v_idx_start, torch::Tensor v_idx_len,
    torch::Tensor X_tree_R, torch::Tensor X_tree_r,
    torch::Tensor inertia_mat,
    torch::Tensor q_min_t, torch::Tensor q_max_t,
    torch::Tensor k_limit_t, torch::Tensor b_limit_t, torch::Tensor damping_t,
    torch::Tensor actuated_q_idx, torch::Tensor actuated_v_idx,
    torch::Tensor effort_limits_t,
    torch::Tensor contact_body_idx, torch::Tensor contact_local_pos,
    int N, int nb, int nq, int nv, int nu, int nc,
    int has_effort_limits,
    float dt, float gravity,
    float kp, float kd, float action_scale, float action_clip,
    float contact_k_normal, float contact_b_normal, float contact_mu,
    float contact_slip_eps, float contact_ground_z
) {
    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    auto opts_i = torch::TensorOptions().dtype(torch::kInt32).device(q.device());
    auto q_new = torch::zeros({N, nq}, opts_f);
    auto qdot_new = torch::zeros({N, nv}, opts_f);
    auto X_world_R_out = torch::zeros({N, nb, 3, 3}, opts_f);
    auto X_world_r_out = torch::zeros({N, nb, 3}, opts_f);
    auto v_bodies_out = torch::zeros({N, nb, 6}, opts_f);
    auto contact_mask_out = torch::zeros({N, nc}, opts_i);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    physics_step_crba_kernel<<<blocks, threads>>>(
        q.data_ptr<float>(), qdot.data_ptr<float>(), actions.data_ptr<float>(),
        joint_type.data_ptr<int>(), joint_axis.data_ptr<float>(),
        parent_idx.data_ptr<int>(), q_idx_start.data_ptr<int>(), q_idx_len.data_ptr<int>(),
        v_idx_start.data_ptr<int>(), v_idx_len.data_ptr<int>(),
        X_tree_R.data_ptr<float>(), X_tree_r.data_ptr<float>(),
        inertia_mat.data_ptr<float>(),
        q_min_t.data_ptr<float>(), q_max_t.data_ptr<float>(),
        k_limit_t.data_ptr<float>(), b_limit_t.data_ptr<float>(), damping_t.data_ptr<float>(),
        actuated_q_idx.data_ptr<int>(), actuated_v_idx.data_ptr<int>(),
        effort_limits_t.data_ptr<float>(),
        contact_body_idx.data_ptr<float>(), contact_local_pos.data_ptr<float>(),
        N, nb, nq, nv, nu, nc, has_effort_limits,
        dt, gravity, kp, kd, action_scale, action_clip,
        contact_k_normal, contact_b_normal, contact_mu, contact_slip_eps, contact_ground_z,
        q_new.data_ptr<float>(), qdot_new.data_ptr<float>(),
        X_world_R_out.data_ptr<float>(), X_world_r_out.data_ptr<float>(),
        v_bodies_out.data_ptr<float>(), contact_mask_out.data_ptr<int>()
    );
    return {q_new, qdot_new, X_world_R_out, X_world_r_out, v_bodies_out, contact_mask_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("physics_step", &physics_step, "Batched physics step ABA (CUDA)");
    m.def("physics_step_crba", &physics_step_crba, "Batched physics step CRBA (CUDA)");
    m.def("fk_only", &fk_only, "FK only (CUDA)");
}
