"""Temporary quantitative probe; run from repository root."""
import json
import numpy as np
from tests.validation.rigid_body.models import build_double_pendulum, build_quadruped, G
from tests.validation.rigid_body.test_dynamics_vs_mujoco import _run_ours
from tests.validation.rigid_body.test_contact_vs_mujoco import _run_cpu
from physics.solvers.pgs_split_impulse import PGSSplitImpulseSolver
from physics.geometry import BoxShape, CylinderShape
from physics.gjk_epa import gjk_epa_query, gjk_distance
from tests.unit.collision.test_convex_margin import _box_hull, _pose

def emit(kind, **values):
    print(json.dumps(dict(kind=kind, **values)), flush=True)

model, _ = build_double_pendulum()
tree = model.tree
def energy(q, v):
    xs = tree.forward_kinematics(q)
    pe = sum(b.inertia.mass * G * (xs[b.index].R @ b.inertia.com + xs[b.index].r)[2] for b in tree.bodies)
    return float(.5 * v @ tree.crba(q) @ v + pe)
q0 = np.array([np.pi / 2, np.pi / 4])
v0 = np.zeros(2)
e0 = energy(q0, v0)
scale = e0 - energy(np.zeros(2), v0)
for dt in [4e-4, 2e-4, 1e-4]:
    qs, vs = _run_ours(model, q0, v0, round(2 / dt) + 1, dt=dt)
    es = np.array([energy(q, v) for q, v in zip(qs, vs)])
    emit('energy', dt=dt, e0=e0, scale=scale, endpoint_j=float(es[-1]-e0), max_j=float(max(abs(es-e0))), old_ratio=float(abs(es[-2]-e0)/abs(e0)), max_ratio=float(max(abs(es-e0))/scale))
for shape in [CylinderShape(.05, .1), _box_hull(.05)]:
    for pen in [1e-5, 1e-4, 1e-3, 3e-3]:
        box = BoxShape((.1, .1, .1))
        a, b = _pose([0, 0, 0]), _pose([0, 0, .1-pen])
        hit = gjk_epa_query(box, a, shape, b, margin=.001)
        distance = gjk_distance(box, a, shape, b, margin=.001)
        emit('geometry', pair=type(shape).__name__, penetration=pen, depth=None if hit is None else hit.depth, distance=None if distance is None else distance[0])
model, xml = build_quadruped(contact=True)
q, v = model.tree.default_state()
q[6] = .45
for name, solver in [('admm', None), ('pgs', PGSSplitImpulseSolver(max_iter=60, erp=.8, slop=.005))]:
    z, vz, qs, vs = _run_cpu(model, q, v, 5000, solver=solver)
    emit('settling', solver=name, mean_z=float(z[-500:].mean()), min_z=float(z.min()), std_z=float(z[-500:].std()), mean_abs_vz=float(abs(vz[-500:]).mean()), max_joint=float(abs(qs[-500:, 7:]).max()))
try:
    from tests.validation.rigid_body.test_contact_vs_mujoco import _run_mujoco_quad
    z, vz, _, _ = _run_mujoco_quad(xml, .45, 5000)
    emit('settling', solver='mujoco', mean_z=float(z[-500:].mean()), mean_abs_vz=float(abs(vz[-500:]).mean()))
except (ImportError, NameError) as exc:
    emit('dependency', error=str(exc))
