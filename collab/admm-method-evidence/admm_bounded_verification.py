"""Diagnose the unchanged ADMM backend, not a replacement solver.

Run from the repository root: python collab/admm-method-evidence/admm_bounded_verification.py
The independent SLSQP oracle is used only on frozen tiny problems and certified
against cone KKT conditions. Its result is never fed back to a simulation.
Exit zero means the investigation and reference checks completed, NOT that the
production solver satisfies the recorded contracts. --gpu requires real CUDA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import scipy
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from physics.cpu_engine import CpuEngine
from physics.geometry import SphereShape
from physics.solvers.admm_qp import ADMMQPSolver
from physics.solvers.pgs_solver import ContactConstraint
from tests.integration.test_cpu_engine_shapes import _make_single_body_engine, _sphere_inertia


class Trace(ADMMQPSolver):
    """Record assembly and iteration count, preserving every production update."""
    def _factor_ARrho(self, ar, rho, n):
        self.Q = ar.copy()
        return super()._factor_ARrho(ar, rho, n)

    def _project_all(self, z, contacts, offsets):
        self.iterations += 1
        return super()._project_all(z, contacts, offsets)

    def solve(self, contacts, tree, q, qdot, a_u, dt):
        self.iterations = 0
        f, J = super().solve(contacts, tree, q, qdot, a_u, dt)
        if contacts:
            # These diagnostics deliberately use constant d=.9 (no sigmoid oracle).
            assert self.solimp[:2] == (.9, .9)
            tc = max(self.solref[0], 2*dt)
            b = 2/(.9*tc)
            k = 1/(.9**2*tc**2*self.solref[1]**2)
            self.c = -b*(J @ qdot) - J @ a_u
            offset = 0
            for contact in contacts:
                self.c[offset] += k*.9*contact.depth
                offset += contact.condim
            self.blocks = [(c.condim, c.mu) for c in contacts]
            self.force = f.copy()
        return f, J


def contact(point, condim=3, mu=1.):
    return ContactConstraint(0, -1, np.array(point, dtype=float), np.array([0.,0.,1.]),
                             np.zeros(3), np.zeros(3), 0., mu, condim=condim)


def cone_conditions(Q, c, f, blocks):
    """Independent feasibility + normal-cone stationarity, no projection routine.

    Boundary/zero classification is used on O(1) frozen fixtures, not trajectories.
    Normal convention: grad + normal = 0, normal points out of the feasible cone.
    """
    grad = Q @ f-c
    feasibility, stationarity = 0., 0.
    offset = 0
    for dim, mu in blocks:
        x, g = f[offset:offset+dim], grad[offset:offset+dim]
        feasibility = max(feasibility, -x[0])
        if dim == 1:
            defect = abs(g[0]) if x[0] > 1e-8 else max(0., -g[0])
        else:
            r = np.linalg.norm(x[1:])
            feasibility = max(feasibility, r-mu*x[0])
            if np.linalg.norm(x) <= 1e-8:
                defect = max(0., mu*np.linalg.norm(g[1:])-g[0])
            elif mu*x[0]-r > 1e-8:
                defect = np.linalg.norm(g)
            else:
                outward = np.r_[-mu, x[1:]/r]
                multiplier = max(0., -g @ outward/(outward @ outward))
                defect = np.linalg.norm(g+multiplier*outward)
        stationarity = max(stationarity, float(defect))
        offset += dim
    return dict(feasibility=float(feasibility), stationarity=float(stationarity))


def oracle(Q, c, blocks):
    """Two independent initial guesses; certify, do not trust success flag alone."""
    np.testing.assert_allclose(Q, Q.T, atol=1e-13, rtol=0)
    assert np.linalg.eigvalsh(Q)[0] > 0
    constraints, bounds = [], []
    offset = 0
    for dim, mu in blocks:
        bounds.extend([(0., None)]+[(None, None)]*(dim-1))
        if dim == 3:
            base = offset
            def fun(x, base=base, mu=mu):
                return mu*x[base]-np.linalg.norm(x[base+1:base+3])
            def jac(x, base=base, mu=mu):
                result = np.zeros(len(x))
                result[base] = mu
                tangent = x[base+1:base+3]
                r = np.linalg.norm(tangent)
                if r:
                    result[base+1:base+3] = -tangent/r
                return result
            constraints.append(dict(type='ineq', fun=fun, jac=jac))
        offset += dim
    outputs = []
    for scale in [.3, 2.]:
        initial = np.zeros(len(c))
        index = 0
        for dim, _ in blocks:
            initial[index] = scale
            index += dim
        result = minimize(lambda x: .5*x @ Q @ x-c @ x, initial,
                          jac=lambda x: Q @ x-c, bounds=bounds, constraints=constraints,
                          method='SLSQP', options=dict(ftol=1e-13, maxiter=2000))
        cert = cone_conditions(Q, c, result.x, blocks)
        assert cert['feasibility'] < 1e-8, (result.message, cert)
        assert cert['stationarity'] < 1e-7*(1+np.linalg.norm(c)), (result.message, cert)
        outputs.append(result.x)
    np.testing.assert_allclose(outputs[0], outputs[1], atol=2e-6, rtol=0)
    return outputs[0], cone_conditions(Q, c, outputs[0], blocks)


def solver(**kwargs):
    return Trace(solimp=(.9,.9,.001,.5,2), warmstart=False, tolerance=1e-11, **kwargs)


def frozen():
    rows = []
    for kind in ['normal', 'friction', 'coupled_normal', 'coupled_friction']:
        merged, _ = _make_single_body_engine(SphereShape(.1), 1., _sphere_inertia(1., .1))
        tree = merged.tree
        q, v = tree.default_state()
        friction = 'friction' in kind
        points = [[0,0,-.1]] if 'coupled' not in kind else [[-.1,0,-.1],[.1,0,-.1]]
        contacts = [contact(p, condim=3 if friction else 1) for p in points]
        accel = np.array([0., -8. if friction else 0., -1., 0., 0., 0.])
        # Analytic body inertia and [direction, arm x direction] Jacobian.
        directions = [[0,0,1],[0,1,0],[-1,0,0]] if friction else [[0,0,1]]
        J_exact = np.array([np.r_[d, np.cross(p,d)] for p in points for d in directions])
        A_exact = (J_exact * np.array([1.,1.,1.,250.,250.,250.])) @ J_exact.T
        Q_exact = A_exact+np.diag(np.diag(A_exact)/9.)
        c_exact = -J_exact @ accel
        blocks = [(c.condim, c.mu) for c in contacts]
        reference, cert = oracle(Q_exact, c_exact, blocks)
        if kind == 'normal':
            np.testing.assert_allclose(reference, [.9], atol=1e-8, rtol=0)
        if kind == 'friction':
            np.testing.assert_allclose(reference, [1.8,1.8,0], atol=1e-8, rtol=0)
        for rho in [.1, 1., 10.]:
            for budget in [50, 1000]:
                s = solver(max_iter=budget, rho=rho, adaptive_rho=False)
                f, J = s.solve(contacts, tree, q, v, accel, .001)
                np.testing.assert_allclose(J, J_exact, atol=1e-13, rtol=0)
                np.testing.assert_allclose(s.Q, Q_exact, atol=1e-13, rtol=0)
                np.testing.assert_allclose(s.c, c_exact, atol=1e-13, rtol=0)
                rows.append(dict(kind=kind, rho=rho, budget=budget, iterations=s.iterations,
                                 returned=f.tolist(), reference=reference.tolist(), reference_kkt=cert,
                                 measured_kkt=cone_conditions(s.Q,s.c,f,blocks),
                                 force_max_error=float(np.max(np.abs(f-reference))),
                                 admm_primal=float(np.linalg.norm(f-s._prev_s)),
                                 Q=Q_exact.tolist(), c=c_exact.tolist(),
                                 objective_difference=float(.5*f@s.Q@f-s.c@f-(.5*reference@s.Q@reference-s.c@reference))))
    return rows


def single_diagonal_reference(Q, c, mu):
    """Analytic O(1) / small impulse SOC optimum for equal tangential eigenvalues."""
    np.testing.assert_allclose(Q, np.diag(np.diag(Q)), atol=1e-7, rtol=0)
    np.testing.assert_allclose(Q[1,1], Q[2,2], atol=1e-7, rtol=1e-7)
    unconstrained = c/np.diag(Q)
    if unconstrained[0] >= 0 and np.linalg.norm(unconstrained[1:]) <= mu*unconstrained[0]:
        return unconstrained
    r = np.linalg.norm(c[1:])
    if c[0]+mu*r <= 0:
        return np.zeros(3)
    n = (c[0]+mu*r)/(Q[0,0]+mu*mu*Q[1,1])
    return np.r_[n, mu*n*c[1:]/r]


def trajectories():
    rows = []
    for vx in [0.,1.,5.]:
        for dt in [.001,.0005,.00025]:
            for budget in [50,1000]:
                merged, _ = _make_single_body_engine(SphereShape(.1),1.,_sphere_inertia(1.,.1))
                s = solver(max_iter=budget, rho=1., adaptive_rho=False)
                engine = CpuEngine(merged, solver=s, dt=dt)
                q,v = merged.tree.default_state()
                q[6],v[0],v[2] = .13,vx,-1.
                first = None
                worst = None
                contact_steps = 0
                missed_penetrating_steps = 0
                max_missed_depth = 0.
                min_z, max_force = q[6],0.
                total_impulse=np.zeros(3)
                previous_v = v.copy()
                max_balance_error = 0.
                for step in range(round(.15/dt)):
                    out=engine.step(q,v,np.zeros(6))
                    if engine.query_contacts():
                        assert len(engine.query_contacts()) == 1
                        contact_steps += 1
                        f=s.force[:3]
                        ref=single_diagonal_reference(s.Q,s.c,s.blocks[0][1])
                        error=float(np.max(np.abs(f-ref)))
                        sample=dict(time=step*dt, depth=.1-q[6], force=f.tolist(),
                                    reference_force=ref.tolist(), max_error=error,
                                    Q=s.Q.tolist(),c=s.c.tolist())
                        if worst is None or error > worst['max_error']:
                            worst=sample
                        p_world=s.last_jacobian[:3,:3].T @ f*dt
                        total_impulse += p_world
                        max_force=max(max_force,float(f[0]))
                        if first is None:
                            first=sample
                    elif q[6] < .1:
                        missed_penetrating_steps += 1
                        max_missed_depth=max(max_missed_depth,float(.1-q[6]))
                    q,v=out.q_new,out.qdot_new
                    predicted_delta=total_impulse+np.array([0,0,-9.81*(step+1)*dt])
                    max_balance_error=max(max_balance_error,float(np.max(np.abs(v[:3]-previous_v[:3]-predicted_delta))))
                    min_z=min(min_z,float(q[6]))
                rows.append(dict(kind='drop' if vx==0 else 'oblique',vx=vx,dt=dt,budget=budget,
                                 first=first,worst_frozen_step=worst,contact_steps=contact_steps,
                                 missed_penetrating_steps=missed_penetrating_steps,
                                 max_missed_depth=max_missed_depth,
                                 final_q=q.tolist(),final_v=v.tolist(),
                                 max_penetration=.1-min_z,max_normal_force=max_force,
                                 total_contact_impulse=total_impulse.tolist(),
                                 linear_impulse_balance_error=max_balance_error))
    return rows


def gpu_first_contact():
    import warp as wp
    from physics.gpu_engine import GpuEngine
    assert wp.get_device('cuda:0').is_cuda
    rows=[]
    for vx in [0.,1.]:
        for mass in [1.,50.]:
            for budget in [30,1000]:
                merged,_=_make_single_body_engine(SphereShape(.1),mass,_sphere_inertia(mass,.1))
                e=GpuEngine(merged,num_envs=1,dt=.0002,solver='admm')
                e._admm_iters=budget
                e._admm_warmstart=False
                e._admm_solimp=(.9,.9,.001,.5,2.)
                q,v=merged.tree.default_state()
                q[6],v[0]=.1-1e-5,vx
                e.reset(q)
                wp.copy(e._scratch.qdot,wp.array(v[None].astype(np.float32),device='cuda:0'))
                e.step()
                indices=np.flatnonzero(e._contact_active.numpy()[0])
                assert len(indices)==1
                r=int(indices[0])*3
                sc=e._solver_scratch
                W=sc.W.numpy()[0,r:r+3,r:r+3].astype(float)
                R=sc.admm_R_diag.numpy()[0,r:r+3].astype(float)
                Q=W+np.diag(R)
                c=sc.admm_rhs_const.numpy()[0,r:r+3].astype(float)
                p=sc.lambdas.numpy()[0,r:r+3].astype(float)
                ref=single_diagonal_reference(Q,c,e._static.contact_mu)
                rows.append(dict(vx=vx,mass=mass,budget=budget,Q=Q.tolist(),rhs=c.tolist(),
                                 returned_impulse=p.tolist(),reference_impulse=ref.tolist(),
                                 relative_error=float(np.linalg.norm(p-ref)/np.linalg.norm(ref)),
                                 R_over_W=(R/np.diag(W)).tolist(),intended_R_over_W=1/9))
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',action='store_true')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__),
                scope='diagnosis only; solver intentionally unchanged',frozen=frozen(),trajectories=trajectories())
    if args.gpu:
        result['gpu']=gpu_first_contact()
    Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(dict(output=args.output,frozen_cases=len(result['frozen']),
                         trajectories=len(result['trajectories']),gpu_cases=len(result.get('gpu',[])))))
