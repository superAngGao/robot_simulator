"""
Analytical collision functions for GPU (Warp @wp.func).

Each function computes contact depth, normal, and point for a specific
shape pair. All operate in world frame.

Return convention: (depth, normal, point_x, point_y, point_z, hit)
  - depth > 0 means penetrating
  - normal points from body_j to body_i
  - point is the contact point in world frame
  - hit = 1 if contact, 0 if separated

Shape params packing (float32[4]):
  Sphere:   [radius, 0, 0, 0]
  Box:      [half_x, half_y, half_z, 0]
  Cylinder: [radius, half_length, 0, 0]
  Capsule:  [radius, half_length, 0, 0]

References:
  Ericson (2004) — Real-Time Collision Detection, chapters 4-5.
  Bullet Physics — btBoxBoxDetector, btCapsuleCapsuleCollisionAlgorithm.
"""

import warp as wp

from physics.contact_tolerances import CONTACT_CONVEX_MARGIN

# Shape type constants (must match static_data.py)
SHAPE_NONE = wp.constant(0)
SHAPE_SPHERE = wp.constant(1)
SHAPE_BOX = wp.constant(2)
SHAPE_CYLINDER = wp.constant(3)
SHAPE_CAPSULE = wp.constant(4)
SHAPE_CONVEXHULL = wp.constant(5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@wp.func
def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@wp.func
def closest_point_on_segment(point: wp.vec3, seg_a: wp.vec3, seg_b: wp.vec3) -> wp.vec3:
    """Closest point on line segment [seg_a, seg_b] to a given point."""
    ab = seg_b - seg_a
    ab_sq = wp.dot(ab, ab)
    if ab_sq < 1.0e-12:
        return seg_a
    t = wp.dot(point - seg_a, ab) / ab_sq
    t = _clamp(t, 0.0, 1.0)
    return seg_a + t * ab


@wp.func
def closest_points_segment_segment(
    p1: wp.vec3,
    d1: wp.vec3,
    len1: float,
    p2: wp.vec3,
    d2: wp.vec3,
    len2: float,
) -> wp.vec3:
    """Closest point on segment 1 to segment 2.

    Segment 1: p1 + t1 * d1, t1 in [0, len1]
    Segment 2: p2 + t2 * d2, t2 in [0, len2]

    Returns the closest point on segment 1.
    Reference: Ericson (2004) §5.1.9.
    """
    r = p1 - p2
    a = wp.dot(d1, d1)  # |d1|² (should be ~1 if normalized)
    e = wp.dot(d2, d2)
    f = wp.dot(d2, r)

    if a < 1.0e-12 and e < 1.0e-12:
        # Both degenerate to points
        return p1

    if a < 1.0e-12:
        # Segment 1 is a point
        return p1

    b = wp.dot(d1, d2)
    c = wp.dot(d1, r)

    if e < 1.0e-12:
        # Segment 2 is a point
        t1 = _clamp(-c / a, 0.0, len1)
        return p1 + t1 * d1

    denom = a * e - b * b
    if denom > 1.0e-12:
        t1 = _clamp((b * f - c * e) / denom, 0.0, len1)
    else:
        t1 = 0.0

    # Compute t2 from t1
    t2 = (b * t1 + f) / e
    if t2 < 0.0:
        t2 = 0.0
        t1 = _clamp(-c / a, 0.0, len1)
    elif t2 > len2:
        t2 = len2
        t1 = _clamp((b * len2 - c) / a, 0.0, len1)

    return p1 + t1 * d1


@wp.func
def closest_points_seg_seg_both(
    p1: wp.vec3,
    d1: wp.vec3,
    len1: float,
    p2: wp.vec3,
    d2: wp.vec3,
    len2: float,
) -> wp.vec3:
    """Returns closest point on segment 2 (complement of above)."""
    r = p1 - p2
    a = wp.dot(d1, d1)
    e = wp.dot(d2, d2)
    f = wp.dot(d2, r)

    if a < 1.0e-12 and e < 1.0e-12:
        return p2

    b = wp.dot(d1, d2)
    c = wp.dot(d1, r)

    if e < 1.0e-12:
        return p2

    if a < 1.0e-12:
        t2 = _clamp(f / e, 0.0, len2)
        return p2 + t2 * d2

    denom = a * e - b * b
    if denom > 1.0e-12:
        t1 = _clamp((b * f - c * e) / denom, 0.0, len1)
    else:
        t1 = 0.0

    t2 = (b * t1 + f) / e
    if t2 < 0.0:
        t2 = 0.0
        t1 = _clamp(-c / a, 0.0, len1)
    elif t2 > len2:
        t2 = len2
        t1 = _clamp((b * len2 - c) / a, 0.0, len1)

    return p2 + t2 * d2


# ---------------------------------------------------------------------------
# Ground collision (shape vs z-plane at ground_z)
# Returns: (depth, contact_point, hit)
# Normal is always (0, 0, 1) for ground.
# ---------------------------------------------------------------------------


@wp.func
def sphere_vs_ground(pos: wp.vec3, radius: float, ground_z: float) -> wp.vec3:
    """Returns (depth, contact_z, hit) packed in vec3."""
    lowest_z = pos[2] - radius
    depth = ground_z - lowest_z
    # Return: x=depth, y=hit (1 or 0), z=lowest_z
    if depth > 0.0:
        return wp.vec3(depth, 1.0, ground_z)
    return wp.vec3(0.0, 0.0, 0.0)


@wp.func
def capsule_vs_ground(
    pos: wp.vec3, R: wp.mat33, radius: float, half_length: float, ground_z: float
) -> wp.vec3:
    """Capsule = sphere-swept segment along local Z.

    Returns (depth, hit, contact_z) packed in vec3.
    Contact point xy = endpoint xy of lowest hemisphere.
    """
    # Two endpoints of capsule axis in world frame
    local_axis = wp.vec3(0.0, 0.0, 1.0)
    axis_world = R * local_axis  # world-frame capsule direction
    ep_a = pos + half_length * axis_world  # top endpoint
    ep_b = pos - half_length * axis_world  # bottom endpoint

    # Pick the lower endpoint
    if ep_a[2] < ep_b[2]:
        low_pt = ep_a
    else:
        low_pt = ep_b

    lowest_z = low_pt[2] - radius
    depth = ground_z - lowest_z
    if depth > 0.0:
        return wp.vec3(depth, 1.0, ground_z)
    return wp.vec3(0.0, 0.0, 0.0)


@wp.func
def _capsule_ground_contact_point(pos: wp.vec3, R: wp.mat33, half_length: float) -> wp.vec3:
    """Returns the xy position of the lower capsule endpoint."""
    axis_world = R * wp.vec3(0.0, 0.0, 1.0)
    ep_a = pos + half_length * axis_world
    ep_b = pos - half_length * axis_world
    if ep_a[2] < ep_b[2]:
        return ep_a
    return ep_b


@wp.func
def box_vs_ground(pos: wp.vec3, R: wp.mat33, hx: float, hy: float, hz: float, ground_z: float) -> wp.vec3:
    """Box support point in -Z direction gives the lowest corner.

    Returns (depth, hit, _) packed in vec3.
    """
    # Support point in -Z: for each local axis, pick sign that minimizes world Z
    # lowest_z = pos[2] - |R[2,0]|*hx - |R[2,1]|*hy - |R[2,2]|*hz
    lowest_z = pos[2] - wp.abs(R[2, 0]) * hx - wp.abs(R[2, 1]) * hy - wp.abs(R[2, 2]) * hz
    depth = ground_z - lowest_z
    if depth > 0.0:
        return wp.vec3(depth, 1.0, ground_z)
    return wp.vec3(0.0, 0.0, 0.0)


@wp.func
def _box_ground_contact_point(pos: wp.vec3, R: wp.mat33, hx: float, hy: float, hz: float) -> wp.vec3:
    """Returns the world position of the box's lowest point (support in -Z)."""
    # Local direction: R^T @ [0,0,-1] = -R[*,2]
    # Sign for each half-extent: -sign(R_col_i . [0,0,1]) = -sign(R[2,i])
    sx = hx
    if R[2, 0] > 0.0:
        sx = -hx
    sy = hy
    if R[2, 1] > 0.0:
        sy = -hy
    sz = hz
    if R[2, 2] > 0.0:
        sz = -hz
    local_pt = wp.vec3(sx, sy, sz)
    return pos + R * local_pt


@wp.func
def cylinder_vs_ground(
    pos: wp.vec3, R: wp.mat33, radius: float, half_length: float, ground_z: float
) -> wp.vec3:
    """Cylinder = disk + segment along local Z.

    Support in -Z = segment endpoint (lower) + disk radius projected onto XY plane.
    """
    axis_world = R * wp.vec3(0.0, 0.0, 1.0)

    # Pick lower endpoint along axis
    if axis_world[2] > 0.0:
        seg_pt = pos - half_length * axis_world  # bottom
    else:
        seg_pt = pos + half_length * axis_world

    # Disk radius contribution: project -Z onto disk plane, normalize, scale by radius
    # Disk plane normal = axis_world
    # Component of -Z perpendicular to axis:
    neg_z = wp.vec3(0.0, 0.0, -1.0)
    perp = neg_z - wp.dot(neg_z, axis_world) * axis_world
    perp_len = wp.length(perp)

    lowest_z = seg_pt[2]
    if perp_len > 1.0e-8:
        disk_offset = (perp / perp_len) * radius
        lowest_z = seg_pt[2] + disk_offset[2]  # disk_offset[2] should be negative
    else:
        # Axis is vertical, disk is horizontal — any rim point has same Z
        lowest_z = seg_pt[2]

    depth = ground_z - lowest_z
    if depth > 0.0:
        return wp.vec3(depth, 1.0, ground_z)
    return wp.vec3(0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Body-body collision: Tier 1
# Returns: (depth, normal, point, hit)
# Normal points from body_j to body_i.
# ---------------------------------------------------------------------------


@wp.func
def sphere_sphere(
    pos_i: wp.vec3,
    r_i: float,
    pos_j: wp.vec3,
    r_j: float,
) -> wp.vec3:
    """Returns (depth, hit, _) in vec3. Normal = (pos_i - pos_j) / dist."""
    diff = pos_i - pos_j
    dist = wp.length(diff)
    overlap = (r_i + r_j) - dist
    if overlap > 0.0 and dist > 1.0e-10:
        return wp.vec3(overlap, 1.0, dist)
    return wp.vec3(0.0, 0.0, 0.0)


@wp.func
def sphere_sphere_normal(pos_i: wp.vec3, pos_j: wp.vec3) -> wp.vec3:
    """Normal from j to i."""
    diff = pos_i - pos_j
    dist = wp.length(diff)
    if dist > 1.0e-10:
        return diff / dist
    return wp.vec3(0.0, 0.0, 1.0)


@wp.func
def sphere_capsule(
    pos_sphere: wp.vec3,
    r_sphere: float,
    pos_cap: wp.vec3,
    R_cap: wp.mat33,
    r_cap: float,
    hl_cap: float,
) -> wp.vec3:
    """Sphere vs capsule. Returns (depth, hit, _) in vec3."""
    # Capsule axis endpoints
    axis = R_cap * wp.vec3(0.0, 0.0, 1.0)
    cap_a = pos_cap + hl_cap * axis
    cap_b = pos_cap - hl_cap * axis

    # Closest point on capsule segment to sphere center
    closest = closest_point_on_segment(pos_sphere, cap_a, cap_b)
    diff = pos_sphere - closest
    dist = wp.length(diff)
    overlap = (r_sphere + r_cap) - dist

    if overlap > 0.0 and dist > 1.0e-10:
        return wp.vec3(overlap, 1.0, dist)
    return wp.vec3(0.0, 0.0, 0.0)


@wp.func
def sphere_capsule_normal_point(
    pos_sphere: wp.vec3,
    r_sphere: float,
    pos_cap: wp.vec3,
    R_cap: wp.mat33,
    r_cap: float,
    hl_cap: float,
) -> wp.vec3:
    """Returns normal (from capsule to sphere) and contact point on capsule surface."""
    axis = R_cap * wp.vec3(0.0, 0.0, 1.0)
    cap_a = pos_cap + hl_cap * axis
    cap_b = pos_cap - hl_cap * axis
    closest = closest_point_on_segment(pos_sphere, cap_a, cap_b)
    diff = pos_sphere - closest
    dist = wp.length(diff)
    if dist > 1.0e-10:
        return diff / dist
    return wp.vec3(0.0, 0.0, 1.0)


@wp.func
def capsule_capsule(
    pos_i: wp.vec3,
    R_i: wp.mat33,
    r_i: float,
    hl_i: float,
    pos_j: wp.vec3,
    R_j: wp.mat33,
    r_j: float,
    hl_j: float,
) -> wp.vec3:
    """Capsule vs capsule. Returns (depth, hit, dist) in vec3."""
    axis_i = R_i * wp.vec3(0.0, 0.0, 1.0)
    axis_j = R_j * wp.vec3(0.0, 0.0, 1.0)

    # Segment i: pos_i - hl_i*axis_i to pos_i + hl_i*axis_i
    seg_i_start = pos_i - hl_i * axis_i

    # Segment j: pos_j - hl_j*axis_j to pos_j + hl_j*axis_j
    seg_j_start = pos_j - hl_j * axis_j

    # Closest points between the two segments
    pt_i = closest_points_segment_segment(
        seg_i_start,
        axis_i,
        2.0 * hl_i,
        seg_j_start,
        axis_j,
        2.0 * hl_j,
    )
    pt_j = closest_points_seg_seg_both(
        seg_i_start,
        axis_i,
        2.0 * hl_i,
        seg_j_start,
        axis_j,
        2.0 * hl_j,
    )

    diff = pt_i - pt_j
    dist = wp.length(diff)
    overlap = (r_i + r_j) - dist

    if overlap > 0.0 and dist > 1.0e-10:
        return wp.vec3(overlap, 1.0, dist)
    return wp.vec3(0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Body-body collision: Tier 2
# ---------------------------------------------------------------------------


@wp.func
def sphere_box(
    pos_sphere: wp.vec3,
    r_sphere: float,
    pos_box: wp.vec3,
    R_box: wp.mat33,
    hx: float,
    hy: float,
    hz: float,
) -> wp.vec3:
    """Sphere vs OBB. Returns (depth, hit, dist) in vec3.

    Reference: Ericson (2004) §5.2.5.
    """
    # Transform sphere center to box local frame
    d = pos_sphere - pos_box
    Rt = wp.transpose(R_box)
    local = Rt * d  # sphere center in box frame

    # Clamp to box
    cx = _clamp(local[0], -hx, hx)
    cy = _clamp(local[1], -hy, hy)
    cz = _clamp(local[2], -hz, hz)
    clamped = wp.vec3(cx, cy, cz)

    diff = local - clamped
    dist = wp.length(diff)

    if dist < 1.0e-10:
        # Sphere center is inside box — find minimum penetration axis
        dx = hx - wp.abs(local[0])
        dy = hy - wp.abs(local[1])
        dz = hz - wp.abs(local[2])
        min_d = dx
        if dy < min_d:
            min_d = dy
        if dz < min_d:
            min_d = dz
        depth = r_sphere + min_d
        return wp.vec3(depth, 1.0, 0.0)

    overlap = r_sphere - dist
    if overlap > 0.0:
        return wp.vec3(overlap, 1.0, dist)
    return wp.vec3(0.0, 0.0, 0.0)


@wp.func
def sphere_box_normal(
    pos_sphere: wp.vec3,
    r_sphere: float,
    pos_box: wp.vec3,
    R_box: wp.mat33,
    hx: float,
    hy: float,
    hz: float,
) -> wp.vec3:
    """Normal from box to sphere."""
    d = pos_sphere - pos_box
    Rt = wp.transpose(R_box)
    local = Rt * d

    cx = _clamp(local[0], -hx, hx)
    cy = _clamp(local[1], -hy, hy)
    cz = _clamp(local[2], -hz, hz)
    clamped = wp.vec3(cx, cy, cz)

    diff = local - clamped
    dist = wp.length(diff)

    if dist < 1.0e-10:
        # Inside: use minimum penetration axis
        dx = hx - wp.abs(local[0])
        dy = hy - wp.abs(local[1])
        dz = hz - wp.abs(local[2])
        if dx <= dy and dx <= dz:
            if local[0] > 0.0:
                return R_box * wp.vec3(1.0, 0.0, 0.0)
            return R_box * wp.vec3(-1.0, 0.0, 0.0)
        elif dy <= dz:
            if local[1] > 0.0:
                return R_box * wp.vec3(0.0, 1.0, 0.0)
            return R_box * wp.vec3(0.0, -1.0, 0.0)
        else:
            if local[2] > 0.0:
                return R_box * wp.vec3(0.0, 0.0, 1.0)
            return R_box * wp.vec3(0.0, 0.0, -1.0)

    # Normal in local frame, transform to world
    n_local = diff / dist
    return R_box * n_local


# ---------------------------------------------------------------------------
# Body-body collision: Tier 3 — Box vs Box (OBB-OBB SAT)
# ---------------------------------------------------------------------------
# SAT tests 15 potential separating axes:
#   - 6 face normals  (3 from A, 3 from B)
#   - 9 edge cross products (A_i × B_j)
# The axis with minimum positive penetration is the contact normal.
#
# Return convention:
#   box_box()         → (depth, hit, _) packed in vec3
#   box_box_normal()  → world-frame unit normal from B to A
#   box_box_contact() → world-frame contact point
#
# Reference: Ericson (2004) §4.4, Gottschalk et al. (1996) OBBTree.
# ---------------------------------------------------------------------------


@wp.func
def _sat_face_axis(
    t_dot_axis: float,
    ra: float,
    rb: float,
) -> float:
    """Compute penetration depth for a single SAT face axis.

    Returns penetration (positive = overlapping) or a large negative
    value if separated on this axis.
    """
    sep = wp.abs(t_dot_axis)
    pen = ra + rb - sep
    return pen


@wp.func
def _sat_edge_axis_pen(
    t: wp.vec3,
    axis: wp.vec3,
    R_a: wp.mat33,
    R_b: wp.mat33,
    ha: wp.vec3,
    hb: wp.vec3,
) -> float:
    """Compute penetration for an edge-edge cross product axis.

    Projects both OBBs onto *axis* and returns penetration depth.
    Axis must be unit length.
    """
    # Project half-extents of A onto axis
    ra = (
        ha[0] * wp.abs(wp.dot(wp.vec3(R_a[0, 0], R_a[1, 0], R_a[2, 0]), axis))
        + ha[1] * wp.abs(wp.dot(wp.vec3(R_a[0, 1], R_a[1, 1], R_a[2, 1]), axis))
        + ha[2] * wp.abs(wp.dot(wp.vec3(R_a[0, 2], R_a[1, 2], R_a[2, 2]), axis))
    )
    rb = (
        hb[0] * wp.abs(wp.dot(wp.vec3(R_b[0, 0], R_b[1, 0], R_b[2, 0]), axis))
        + hb[1] * wp.abs(wp.dot(wp.vec3(R_b[0, 1], R_b[1, 1], R_b[2, 1]), axis))
        + hb[2] * wp.abs(wp.dot(wp.vec3(R_b[0, 2], R_b[1, 2], R_b[2, 2]), axis))
    )
    sep = wp.abs(wp.dot(t, axis))
    return ra + rb - sep


@wp.func
def box_box(
    pos_a: wp.vec3,
    R_a: wp.mat33,
    ha_x: float,
    ha_y: float,
    ha_z: float,
    pos_b: wp.vec3,
    R_b: wp.mat33,
    hb_x: float,
    hb_y: float,
    hb_z: float,
) -> wp.vec3:
    """OBB vs OBB via SAT (15 axes). Returns (depth, hit, _) in vec3.

    Reference: Ericson (2004) §4.4 — OBB intersection test.
    """
    t = pos_b - pos_a  # center offset (world frame)

    # Column vectors of each rotation matrix (box local axes in world)
    ax0 = wp.vec3(R_a[0, 0], R_a[1, 0], R_a[2, 0])
    ax1 = wp.vec3(R_a[0, 1], R_a[1, 1], R_a[2, 1])
    ax2 = wp.vec3(R_a[0, 2], R_a[1, 2], R_a[2, 2])
    bx0 = wp.vec3(R_b[0, 0], R_b[1, 0], R_b[2, 0])
    bx1 = wp.vec3(R_b[0, 1], R_b[1, 1], R_b[2, 1])
    bx2 = wp.vec3(R_b[0, 2], R_b[1, 2], R_b[2, 2])

    ha = wp.vec3(ha_x, ha_y, ha_z)
    hb = wp.vec3(hb_x, hb_y, hb_z)

    # Precompute R_rel[i][j] = dot(ax_i, bx_j) and abs version
    # (Ericson §4.4 optimisation — avoids recomputing per-axis projections)
    r00 = wp.dot(ax0, bx0)
    r01 = wp.dot(ax0, bx1)
    r02 = wp.dot(ax0, bx2)
    r10 = wp.dot(ax1, bx0)
    r11 = wp.dot(ax1, bx1)
    r12 = wp.dot(ax1, bx2)
    r20 = wp.dot(ax2, bx0)
    r21 = wp.dot(ax2, bx1)
    r22 = wp.dot(ax2, bx2)

    ar00 = wp.abs(r00) + 1.0e-8
    ar01 = wp.abs(r01) + 1.0e-8
    ar02 = wp.abs(r02) + 1.0e-8
    ar10 = wp.abs(r10) + 1.0e-8
    ar11 = wp.abs(r11) + 1.0e-8
    ar12 = wp.abs(r12) + 1.0e-8
    ar20 = wp.abs(r20) + 1.0e-8
    ar21 = wp.abs(r21) + 1.0e-8
    ar22 = wp.abs(r22) + 1.0e-8

    # t projected onto A's local axes
    ta0 = wp.dot(t, ax0)
    ta1 = wp.dot(t, ax1)
    ta2 = wp.dot(t, ax2)

    min_pen = 1.0e30  # large sentinel

    # --- 6 face axes ---
    # A's face 0 (ax0)
    pen = _sat_face_axis(ta0, ha[0], hb[0] * ar00 + hb[1] * ar01 + hb[2] * ar02)
    if pen < 0.0:
        return wp.vec3(0.0, 0.0, 0.0)
    if pen < min_pen:
        min_pen = pen

    # A's face 1 (ax1)
    pen = _sat_face_axis(ta1, ha[1], hb[0] * ar10 + hb[1] * ar11 + hb[2] * ar12)
    if pen < 0.0:
        return wp.vec3(0.0, 0.0, 0.0)
    if pen < min_pen:
        min_pen = pen

    # A's face 2 (ax2)
    pen = _sat_face_axis(ta2, ha[2], hb[0] * ar20 + hb[1] * ar21 + hb[2] * ar22)
    if pen < 0.0:
        return wp.vec3(0.0, 0.0, 0.0)
    if pen < min_pen:
        min_pen = pen

    # B's face 0 (bx0): t projected onto bx0
    tb0 = wp.dot(t, bx0)
    pen = _sat_face_axis(tb0, ha[0] * ar00 + ha[1] * ar10 + ha[2] * ar20, hb[0])
    if pen < 0.0:
        return wp.vec3(0.0, 0.0, 0.0)
    if pen < min_pen:
        min_pen = pen

    # B's face 1 (bx1)
    tb1 = wp.dot(t, bx1)
    pen = _sat_face_axis(tb1, ha[0] * ar01 + ha[1] * ar11 + ha[2] * ar21, hb[1])
    if pen < 0.0:
        return wp.vec3(0.0, 0.0, 0.0)
    if pen < min_pen:
        min_pen = pen

    # B's face 2 (bx2)
    tb2 = wp.dot(t, bx2)
    pen = _sat_face_axis(tb2, ha[0] * ar02 + ha[1] * ar12 + ha[2] * ar22, hb[2])
    if pen < 0.0:
        return wp.vec3(0.0, 0.0, 0.0)
    if pen < min_pen:
        min_pen = pen

    # --- 9 edge-edge axes (A_i × B_j) ---
    # Each cross product may be degenerate (parallel edges → zero axis).
    # Skip degenerate axes (they are redundant with face axes).

    # A0 × B0
    cross = wp.cross(ax0, bx0)
    cl = wp.length(cross)
    if cl > 1.0e-6:
        axis = cross / cl
        pen = _sat_edge_axis_pen(t, axis, R_a, R_b, ha, hb)
        if pen < 0.0:
            return wp.vec3(0.0, 0.0, 0.0)
        if pen < min_pen:
            min_pen = pen

    # A0 × B1
    cross = wp.cross(ax0, bx1)
    cl = wp.length(cross)
    if cl > 1.0e-6:
        axis = cross / cl
        pen = _sat_edge_axis_pen(t, axis, R_a, R_b, ha, hb)
        if pen < 0.0:
            return wp.vec3(0.0, 0.0, 0.0)
        if pen < min_pen:
            min_pen = pen

    # A0 × B2
    cross = wp.cross(ax0, bx2)
    cl = wp.length(cross)
    if cl > 1.0e-6:
        axis = cross / cl
        pen = _sat_edge_axis_pen(t, axis, R_a, R_b, ha, hb)
        if pen < 0.0:
            return wp.vec3(0.0, 0.0, 0.0)
        if pen < min_pen:
            min_pen = pen

    # A1 × B0
    cross = wp.cross(ax1, bx0)
    cl = wp.length(cross)
    if cl > 1.0e-6:
        axis = cross / cl
        pen = _sat_edge_axis_pen(t, axis, R_a, R_b, ha, hb)
        if pen < 0.0:
            return wp.vec3(0.0, 0.0, 0.0)
        if pen < min_pen:
            min_pen = pen

    # A1 × B1
    cross = wp.cross(ax1, bx1)
    cl = wp.length(cross)
    if cl > 1.0e-6:
        axis = cross / cl
        pen = _sat_edge_axis_pen(t, axis, R_a, R_b, ha, hb)
        if pen < 0.0:
            return wp.vec3(0.0, 0.0, 0.0)
        if pen < min_pen:
            min_pen = pen

    # A1 × B2
    cross = wp.cross(ax1, bx2)
    cl = wp.length(cross)
    if cl > 1.0e-6:
        axis = cross / cl
        pen = _sat_edge_axis_pen(t, axis, R_a, R_b, ha, hb)
        if pen < 0.0:
            return wp.vec3(0.0, 0.0, 0.0)
        if pen < min_pen:
            min_pen = pen

    # A2 × B0
    cross = wp.cross(ax2, bx0)
    cl = wp.length(cross)
    if cl > 1.0e-6:
        axis = cross / cl
        pen = _sat_edge_axis_pen(t, axis, R_a, R_b, ha, hb)
        if pen < 0.0:
            return wp.vec3(0.0, 0.0, 0.0)
        if pen < min_pen:
            min_pen = pen

    # A2 × B1
    cross = wp.cross(ax2, bx1)
    cl = wp.length(cross)
    if cl > 1.0e-6:
        axis = cross / cl
        pen = _sat_edge_axis_pen(t, axis, R_a, R_b, ha, hb)
        if pen < 0.0:
            return wp.vec3(0.0, 0.0, 0.0)
        if pen < min_pen:
            min_pen = pen

    # A2 × B2
    cross = wp.cross(ax2, bx2)
    cl = wp.length(cross)
    if cl > 1.0e-6:
        axis = cross / cl
        pen = _sat_edge_axis_pen(t, axis, R_a, R_b, ha, hb)
        if pen < 0.0:
            return wp.vec3(0.0, 0.0, 0.0)
        if pen < min_pen:
            min_pen = pen

    if min_pen > 0.0 and min_pen < 1.0e29:
        return wp.vec3(min_pen, 1.0, 0.0)
    return wp.vec3(0.0, 0.0, 0.0)


@wp.func
def box_box_normal(
    pos_a: wp.vec3,
    R_a: wp.mat33,
    ha_x: float,
    ha_y: float,
    ha_z: float,
    pos_b: wp.vec3,
    R_b: wp.mat33,
    hb_x: float,
    hb_y: float,
    hb_z: float,
) -> wp.vec3:
    """Contact normal (from B to A) for OBB-OBB via SAT minimum axis.

    Re-evaluates all 15 axes to find the minimum penetration axis.
    Returns a unit normal in world frame pointing from B toward A.
    """
    t = pos_b - pos_a

    ax0 = wp.vec3(R_a[0, 0], R_a[1, 0], R_a[2, 0])
    ax1 = wp.vec3(R_a[0, 1], R_a[1, 1], R_a[2, 1])
    ax2 = wp.vec3(R_a[0, 2], R_a[1, 2], R_a[2, 2])
    bx0 = wp.vec3(R_b[0, 0], R_b[1, 0], R_b[2, 0])
    bx1 = wp.vec3(R_b[0, 1], R_b[1, 1], R_b[2, 1])
    bx2 = wp.vec3(R_b[0, 2], R_b[1, 2], R_b[2, 2])

    ha = wp.vec3(ha_x, ha_y, ha_z)
    hb = wp.vec3(hb_x, hb_y, hb_z)

    r00 = wp.dot(ax0, bx0)
    r01 = wp.dot(ax0, bx1)
    r02 = wp.dot(ax0, bx2)
    r10 = wp.dot(ax1, bx0)
    r11 = wp.dot(ax1, bx1)
    r12 = wp.dot(ax1, bx2)
    r20 = wp.dot(ax2, bx0)
    r21 = wp.dot(ax2, bx1)
    r22 = wp.dot(ax2, bx2)

    ar00 = wp.abs(r00) + 1.0e-8
    ar01 = wp.abs(r01) + 1.0e-8
    ar02 = wp.abs(r02) + 1.0e-8
    ar10 = wp.abs(r10) + 1.0e-8
    ar11 = wp.abs(r11) + 1.0e-8
    ar12 = wp.abs(r12) + 1.0e-8
    ar20 = wp.abs(r20) + 1.0e-8
    ar21 = wp.abs(r21) + 1.0e-8
    ar22 = wp.abs(r22) + 1.0e-8

    ta0 = wp.dot(t, ax0)
    ta1 = wp.dot(t, ax1)
    ta2 = wp.dot(t, ax2)

    min_pen = 1.0e30
    best_axis = wp.vec3(0.0, 0.0, 1.0)

    # --- Face axes ---
    pen = _sat_face_axis(ta0, ha[0], hb[0] * ar00 + hb[1] * ar01 + hb[2] * ar02)
    if pen < min_pen and pen > 0.0:
        min_pen = pen
        best_axis = ax0

    pen = _sat_face_axis(ta1, ha[1], hb[0] * ar10 + hb[1] * ar11 + hb[2] * ar12)
    if pen < min_pen and pen > 0.0:
        min_pen = pen
        best_axis = ax1

    pen = _sat_face_axis(ta2, ha[2], hb[0] * ar20 + hb[1] * ar21 + hb[2] * ar22)
    if pen < min_pen and pen > 0.0:
        min_pen = pen
        best_axis = ax2

    tb0 = wp.dot(t, bx0)
    pen = _sat_face_axis(tb0, ha[0] * ar00 + ha[1] * ar10 + ha[2] * ar20, hb[0])
    if pen < min_pen and pen > 0.0:
        min_pen = pen
        best_axis = bx0

    tb1 = wp.dot(t, bx1)
    pen = _sat_face_axis(tb1, ha[0] * ar01 + ha[1] * ar11 + ha[2] * ar21, hb[1])
    if pen < min_pen and pen > 0.0:
        min_pen = pen
        best_axis = bx1

    tb2 = wp.dot(t, bx2)
    pen = _sat_face_axis(tb2, ha[0] * ar02 + ha[1] * ar12 + ha[2] * ar22, hb[2])
    if pen < min_pen and pen > 0.0:
        min_pen = pen
        best_axis = bx2

    # --- Edge-edge axes ---
    cross = wp.cross(ax0, bx0)
    cl = wp.length(cross)
    if cl > 1.0e-6:
        axis = cross / cl
        pen = _sat_edge_axis_pen(t, axis, R_a, R_b, ha, hb)
        if pen < min_pen and pen > 0.0:
            min_pen = pen
            best_axis = axis

    cross = wp.cross(ax0, bx1)
    cl = wp.length(cross)
    if cl > 1.0e-6:
        axis = cross / cl
        pen = _sat_edge_axis_pen(t, axis, R_a, R_b, ha, hb)
        if pen < min_pen and pen > 0.0:
            min_pen = pen
            best_axis = axis

    cross = wp.cross(ax0, bx2)
    cl = wp.length(cross)
    if cl > 1.0e-6:
        axis = cross / cl
        pen = _sat_edge_axis_pen(t, axis, R_a, R_b, ha, hb)
        if pen < min_pen and pen > 0.0:
            min_pen = pen
            best_axis = axis

    cross = wp.cross(ax1, bx0)
    cl = wp.length(cross)
    if cl > 1.0e-6:
        axis = cross / cl
        pen = _sat_edge_axis_pen(t, axis, R_a, R_b, ha, hb)
        if pen < min_pen and pen > 0.0:
            min_pen = pen
            best_axis = axis

    cross = wp.cross(ax1, bx1)
    cl = wp.length(cross)
    if cl > 1.0e-6:
        axis = cross / cl
        pen = _sat_edge_axis_pen(t, axis, R_a, R_b, ha, hb)
        if pen < min_pen and pen > 0.0:
            min_pen = pen
            best_axis = axis

    cross = wp.cross(ax1, bx2)
    cl = wp.length(cross)
    if cl > 1.0e-6:
        axis = cross / cl
        pen = _sat_edge_axis_pen(t, axis, R_a, R_b, ha, hb)
        if pen < min_pen and pen > 0.0:
            min_pen = pen
            best_axis = axis

    cross = wp.cross(ax2, bx0)
    cl = wp.length(cross)
    if cl > 1.0e-6:
        axis = cross / cl
        pen = _sat_edge_axis_pen(t, axis, R_a, R_b, ha, hb)
        if pen < min_pen and pen > 0.0:
            min_pen = pen
            best_axis = axis

    cross = wp.cross(ax2, bx1)
    cl = wp.length(cross)
    if cl > 1.0e-6:
        axis = cross / cl
        pen = _sat_edge_axis_pen(t, axis, R_a, R_b, ha, hb)
        if pen < min_pen and pen > 0.0:
            min_pen = pen
            best_axis = axis

    cross = wp.cross(ax2, bx2)
    cl = wp.length(cross)
    if cl > 1.0e-6:
        axis = cross / cl
        pen = _sat_edge_axis_pen(t, axis, R_a, R_b, ha, hb)
        if pen < min_pen and pen > 0.0:
            min_pen = pen
            best_axis = axis

    # Orient normal: should point from B to A (i.e. opposite to t = B - A)
    if wp.dot(best_axis, t) > 0.0:
        best_axis = wp.vec3(-best_axis[0], -best_axis[1], -best_axis[2])

    return best_axis


@wp.func
def box_box_contact_point(
    pos_a: wp.vec3,
    R_a: wp.mat33,
    ha_x: float,
    ha_y: float,
    ha_z: float,
    pos_b: wp.vec3,
    R_b: wp.mat33,
    hb_x: float,
    hb_y: float,
    hb_z: float,
    normal: wp.vec3,
) -> wp.vec3:
    """Contact point for OBB-OBB: support midpoint on the contact normal.

    Finds the deepest point on each box along the contact normal and
    returns their midpoint.  This is the single-point approximation;
    multi-point manifold is a separate pass.

    Reference: Ericson (2004) §5.5.7.
    """
    # Support point of A in direction -normal (deepest into B)
    Rt_a = wp.transpose(R_a)
    n_local_a = Rt_a * wp.vec3(-normal[0], -normal[1], -normal[2])
    sx_a = ha_x
    if n_local_a[0] < 0.0:
        sx_a = -ha_x
    sy_a = ha_y
    if n_local_a[1] < 0.0:
        sy_a = -ha_y
    sz_a = ha_z
    if n_local_a[2] < 0.0:
        sz_a = -ha_z
    sup_a = pos_a + R_a * wp.vec3(sx_a, sy_a, sz_a)

    # Support point of B in direction +normal (deepest into A)
    Rt_b = wp.transpose(R_b)
    n_local_b = Rt_b * normal
    sx_b = hb_x
    if n_local_b[0] < 0.0:
        sx_b = -hb_x
    sy_b = hb_y
    if n_local_b[1] < 0.0:
        sy_b = -hb_y
    sz_b = hb_z
    if n_local_b[2] < 0.0:
        sz_b = -hb_z
    sup_b = pos_b + R_b * wp.vec3(sx_b, sy_b, sz_b)

    # Midpoint
    return (sup_a + sup_b) * 0.5


# ---------------------------------------------------------------------------
# Multi-point contact manifold structs and helpers
# ---------------------------------------------------------------------------


@wp.struct
class ClipPoly:
    """Polygon buffer for Sutherland-Hodgman clipping (max 8 vertices).

    A quad (4 verts) clipped by 4 half-planes produces at most 8 vertices.
    """

    v0: wp.vec3
    v1: wp.vec3
    v2: wp.vec3
    v3: wp.vec3
    v4: wp.vec3
    v5: wp.vec3
    v6: wp.vec3
    v7: wp.vec3
    count: wp.int32


@wp.struct
class ContactPolyManifold:
    """Up to 4 contact points from polytope-polytope / polytope-ground contact.

    Generic name — used by box-box S-H face clipping, box/convexhull vs
    ground vertex enumeration, and upcoming cylinder-prism pipelines.
    """

    p0: wp.vec3
    p1: wp.vec3
    p2: wp.vec3
    p3: wp.vec3
    d0: wp.float32
    d1: wp.float32
    d2: wp.float32
    d3: wp.float32
    count: wp.int32


@wp.func
def _clip_poly_get(poly: ClipPoly, i: int) -> wp.vec3:
    """Read vertex i from ClipPoly (0-indexed, 8-branch)."""
    if i == 0:
        return poly.v0
    if i == 1:
        return poly.v1
    if i == 2:
        return poly.v2
    if i == 3:
        return poly.v3
    if i == 4:
        return poly.v4
    if i == 5:
        return poly.v5
    if i == 6:
        return poly.v6
    return poly.v7


@wp.func
def _clip_poly_push(poly: ClipPoly, v: wp.vec3) -> ClipPoly:
    """Append vertex to ClipPoly, return updated struct."""
    idx = poly.count
    if idx == 0:
        poly.v0 = v
    elif idx == 1:
        poly.v1 = v
    elif idx == 2:
        poly.v2 = v
    elif idx == 3:
        poly.v3 = v
    elif idx == 4:
        poly.v4 = v
    elif idx == 5:
        poly.v5 = v
    elif idx == 6:
        poly.v6 = v
    elif idx == 7:
        poly.v7 = v
    poly.count = idx + 1
    return poly


@wp.func
def _manifold_get_point(m: ContactPolyManifold, i: int) -> wp.vec3:
    if i == 0:
        return m.p0
    if i == 1:
        return m.p1
    if i == 2:
        return m.p2
    return m.p3


@wp.func
def _manifold_get_depth(m: ContactPolyManifold, i: int) -> wp.float32:
    if i == 0:
        return m.d0
    if i == 1:
        return m.d1
    if i == 2:
        return m.d2
    return m.d3


@wp.func
def _manifold_set(m: ContactPolyManifold, i: int, p: wp.vec3, d: float) -> ContactPolyManifold:
    """Set point i in manifold, return updated struct."""
    if i == 0:
        m.p0 = p
        m.d0 = d
    elif i == 1:
        m.p1 = p
        m.d1 = d
    elif i == 2:
        m.p2 = p
        m.d2 = d
    elif i == 3:
        m.p3 = p
        m.d3 = d
    return m


# ---------------------------------------------------------------------------
# Box face geometry
# ---------------------------------------------------------------------------


@wp.func
def _box_face_normal_world(R: wp.mat33, face_idx: int) -> wp.vec3:
    """World-frame outward normal for box face_idx (0-5: +X,-X,+Y,-Y,+Z,-Z)."""
    if face_idx == 0:
        return wp.vec3(R[0, 0], R[1, 0], R[2, 0])
    if face_idx == 1:
        return wp.vec3(-R[0, 0], -R[1, 0], -R[2, 0])
    if face_idx == 2:
        return wp.vec3(R[0, 1], R[1, 1], R[2, 1])
    if face_idx == 3:
        return wp.vec3(-R[0, 1], -R[1, 1], -R[2, 1])
    if face_idx == 4:
        return wp.vec3(R[0, 2], R[1, 2], R[2, 2])
    return wp.vec3(-R[0, 2], -R[1, 2], -R[2, 2])


@wp.func
def _box_face_vertices(
    pos: wp.vec3,
    R: wp.mat33,
    hx: float,
    hy: float,
    hz: float,
    face_idx: int,
) -> ClipPoly:
    """4 world-frame vertices of box face face_idx in CCW winding.

    Face indexing: 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z.
    CCW winding is viewed from outside the box (outward normal points at viewer).
    """
    # Local-frame vertices for each face
    l0 = wp.vec3(0.0, 0.0, 0.0)
    l1 = wp.vec3(0.0, 0.0, 0.0)
    l2 = wp.vec3(0.0, 0.0, 0.0)
    l3 = wp.vec3(0.0, 0.0, 0.0)

    if face_idx == 0:  # +X face
        l0 = wp.vec3(hx, -hy, -hz)
        l1 = wp.vec3(hx, hy, -hz)
        l2 = wp.vec3(hx, hy, hz)
        l3 = wp.vec3(hx, -hy, hz)
    elif face_idx == 1:  # -X face
        l0 = wp.vec3(-hx, hy, -hz)
        l1 = wp.vec3(-hx, -hy, -hz)
        l2 = wp.vec3(-hx, -hy, hz)
        l3 = wp.vec3(-hx, hy, hz)
    elif face_idx == 2:  # +Y face
        l0 = wp.vec3(hx, hy, -hz)
        l1 = wp.vec3(-hx, hy, -hz)
        l2 = wp.vec3(-hx, hy, hz)
        l3 = wp.vec3(hx, hy, hz)
    elif face_idx == 3:  # -Y face
        l0 = wp.vec3(-hx, -hy, -hz)
        l1 = wp.vec3(hx, -hy, -hz)
        l2 = wp.vec3(hx, -hy, hz)
        l3 = wp.vec3(-hx, -hy, hz)
    elif face_idx == 4:  # +Z face
        l0 = wp.vec3(-hx, -hy, hz)
        l1 = wp.vec3(hx, -hy, hz)
        l2 = wp.vec3(hx, hy, hz)
        l3 = wp.vec3(-hx, hy, hz)
    else:  # face_idx == 5, -Z face
        l0 = wp.vec3(-hx, hy, -hz)
        l1 = wp.vec3(hx, hy, -hz)
        l2 = wp.vec3(hx, -hy, -hz)
        l3 = wp.vec3(-hx, -hy, -hz)

    poly = ClipPoly()
    poly.v0 = pos + R * l0
    poly.v1 = pos + R * l1
    poly.v2 = pos + R * l2
    poly.v3 = pos + R * l3
    poly.count = 4
    return poly


@wp.func
def _box_find_support_face(R: wp.mat33, direction: wp.vec3) -> int:
    """Find box face whose outward normal has highest dot with direction.

    Returns face index 0-5 (+X,-X,+Y,-Y,+Z,-Z).
    """
    ax0 = wp.vec3(R[0, 0], R[1, 0], R[2, 0])
    ax1 = wp.vec3(R[0, 1], R[1, 1], R[2, 1])
    ax2 = wp.vec3(R[0, 2], R[1, 2], R[2, 2])

    d0 = wp.dot(direction, ax0)
    d1 = wp.dot(direction, ax1)
    d2 = wp.dot(direction, ax2)

    best_dot = d0
    best_idx = 0
    if -d0 > best_dot:
        best_dot = -d0
        best_idx = 1
    if d1 > best_dot:
        best_dot = d1
        best_idx = 2
    if -d1 > best_dot:
        best_dot = -d1
        best_idx = 3
    if d2 > best_dot:
        best_dot = d2
        best_idx = 4
    if -d2 > best_dot:
        best_idx = 5

    return best_idx


@wp.func
def _box_find_incident_face(R: wp.mat33, ref_normal: wp.vec3) -> int:
    """Find box face most anti-aligned with ref_normal (min dot)."""
    ax0 = wp.vec3(R[0, 0], R[1, 0], R[2, 0])
    ax1 = wp.vec3(R[0, 1], R[1, 1], R[2, 1])
    ax2 = wp.vec3(R[0, 2], R[1, 2], R[2, 2])

    d0 = wp.dot(ref_normal, ax0)
    d1 = wp.dot(ref_normal, ax1)
    d2 = wp.dot(ref_normal, ax2)

    best_dot = d0
    best_idx = 0
    if -d0 < best_dot:
        best_dot = -d0
        best_idx = 1
    if d1 < best_dot:
        best_dot = d1
        best_idx = 2
    if -d1 < best_dot:
        best_dot = -d1
        best_idx = 3
    if d2 < best_dot:
        best_dot = d2
        best_idx = 4
    if -d2 < best_dot:
        best_idx = 5

    return best_idx


# ---------------------------------------------------------------------------
# Sutherland-Hodgman polygon clipping
# ---------------------------------------------------------------------------


@wp.func
def _clip_polygon_by_plane(poly: ClipPoly, plane_n: wp.vec3, plane_d: float) -> ClipPoly:
    """Clip polygon against half-plane {x : dot(plane_n, x) <= plane_d}.

    Sutherland-Hodgman: for each edge, classify endpoints and emit
    kept vertices + intersection points.
    """
    out = ClipPoly()
    out.count = 0
    n = poly.count

    for i in range(8):
        if i >= n:
            continue
        curr = _clip_poly_get(poly, i)
        next_idx = (i + 1) % n
        nxt = _clip_poly_get(poly, next_idx)

        d_curr = wp.dot(plane_n, curr) - plane_d
        d_nxt = wp.dot(plane_n, nxt) - plane_d

        if d_curr <= 0.0:
            # Current inside
            if out.count < 8:
                out = _clip_poly_push(out, curr)
            if d_nxt > 0.0:
                # Next outside — add intersection
                denom = d_curr - d_nxt
                if wp.abs(denom) > 1.0e-12:
                    t = d_curr / denom
                    inter = curr + t * (nxt - curr)
                    if out.count < 8:
                        out = _clip_poly_push(out, inter)
        else:
            # Current outside
            if d_nxt <= 0.0:
                # Next inside — add intersection
                denom = d_curr - d_nxt
                if wp.abs(denom) > 1.0e-12:
                    t = d_curr / denom
                    inter = curr + t * (nxt - curr)
                    if out.count < 8:
                        out = _clip_poly_push(out, inter)

    return out


# ---------------------------------------------------------------------------
# Box-box multi-point manifold via face clipping
# ---------------------------------------------------------------------------


@wp.func
def box_box_manifold(
    pos_a: wp.vec3,
    R_a: wp.mat33,
    ha_x: float,
    ha_y: float,
    ha_z: float,
    pos_b: wp.vec3,
    R_b: wp.mat33,
    hb_x: float,
    hb_y: float,
    hb_z: float,
    normal: wp.vec3,
    depth: float,
) -> ContactPolyManifold:
    """Multi-point contact manifold for OBB-OBB collision.

    Given the SAT contact normal (B→A) and penetration depth, produces
    up to 4 contact points via Sutherland-Hodgman face clipping.

    For edge-edge contacts (normal not aligned with any face axis),
    falls back to the single support-midpoint.

    Reference: Ericson (2004) §5.5, Bullet btBoxBoxDetector.
    """
    result = ContactPolyManifold()
    result.count = 0

    # --- Axis type detection ---
    # Check alignment of normal with face axes of A and B
    ax0 = wp.vec3(R_a[0, 0], R_a[1, 0], R_a[2, 0])
    ax1 = wp.vec3(R_a[0, 1], R_a[1, 1], R_a[2, 1])
    ax2 = wp.vec3(R_a[0, 2], R_a[1, 2], R_a[2, 2])

    da0 = wp.abs(wp.dot(normal, ax0))
    da1 = wp.abs(wp.dot(normal, ax1))
    da2 = wp.abs(wp.dot(normal, ax2))
    max_dot_a = da0
    if da1 > max_dot_a:
        max_dot_a = da1
    if da2 > max_dot_a:
        max_dot_a = da2

    bx0 = wp.vec3(R_b[0, 0], R_b[1, 0], R_b[2, 0])
    bx1 = wp.vec3(R_b[0, 1], R_b[1, 1], R_b[2, 1])
    bx2 = wp.vec3(R_b[0, 2], R_b[1, 2], R_b[2, 2])

    db0 = wp.abs(wp.dot(normal, bx0))
    db1 = wp.abs(wp.dot(normal, bx1))
    db2 = wp.abs(wp.dot(normal, bx2))
    max_dot_b = db0
    if db1 > max_dot_b:
        max_dot_b = db1
    if db2 > max_dot_b:
        max_dot_b = db2

    FACE_THRESH = 0.99

    if max_dot_a < FACE_THRESH and max_dot_b < FACE_THRESH:
        # Edge-edge contact: single point (support midpoint)
        cp = box_box_contact_point(
            pos_a,
            R_a,
            ha_x,
            ha_y,
            ha_z,
            pos_b,
            R_b,
            hb_x,
            hb_y,
            hb_z,
            normal,
        )
        result.p0 = cp
        result.d0 = depth
        result.count = 1
        return result

    # --- Face clipping path ---
    # Reference face = box whose face normal aligns better with SAT normal
    # The reference face normal should point in the same direction as `normal`
    # (from B toward A), so for A we look for face aligned with -normal
    # (A's outward face toward B), for B we look for face aligned with +normal.

    ref_pos = pos_a
    ref_R = R_a
    ref_hx = ha_x
    ref_hy = ha_y
    ref_hz = ha_z
    inc_pos = pos_b
    inc_R = R_b
    inc_hx = hb_x
    inc_hy = hb_y
    inc_hz = hb_z
    # Reference face: A's face whose outward normal ≈ -normal (faces toward B)
    neg_n = wp.vec3(-normal[0], -normal[1], -normal[2])
    ref_face = _box_find_support_face(R_a, neg_n)
    ref_fn = _box_face_normal_world(R_a, ref_face)

    if max_dot_b > max_dot_a:
        # B is reference instead
        ref_pos = pos_b
        ref_R = R_b
        ref_hx = hb_x
        ref_hy = hb_y
        ref_hz = hb_z
        inc_pos = pos_a
        inc_R = R_a
        inc_hx = ha_x
        inc_hy = ha_y
        inc_hz = ha_z
        ref_face = _box_find_support_face(R_b, normal)
        ref_fn = _box_face_normal_world(R_b, ref_face)

    # Incident face: face on incident box most anti-aligned with ref normal
    inc_face = _box_find_incident_face(inc_R, ref_fn)

    # Get face polygons in world frame
    ref_poly = _box_face_vertices(ref_pos, ref_R, ref_hx, ref_hy, ref_hz, ref_face)
    inc_poly = _box_face_vertices(inc_pos, inc_R, inc_hx, inc_hy, inc_hz, inc_face)

    # Clip incident polygon against 4 side planes of reference face
    # Side plane for edge (v_i, v_{i+1}): normal = cross(edge, ref_fn), pointing inward
    for ei in range(4):
        ev0 = _clip_poly_get(ref_poly, ei)
        ev1 = _clip_poly_get(ref_poly, (ei + 1) % 4)
        edge = ev1 - ev0
        side_n = wp.normalize(wp.cross(edge, ref_fn))
        side_d = wp.dot(side_n, ev0)
        inc_poly = _clip_polygon_by_plane(inc_poly, side_n, side_d)

    # Project clipped points onto reference plane, filter by depth
    ref_d = wp.dot(ref_fn, _clip_poly_get(ref_poly, 0))

    for ci in range(8):
        if ci >= inc_poly.count:
            continue
        if result.count >= 4:
            continue
        p = _clip_poly_get(inc_poly, ci)
        signed_dist = wp.dot(ref_fn, p) - ref_d
        pt_depth = -signed_dist
        if pt_depth > -1.0e-6:
            # Project point onto reference plane
            contact_pt = p - signed_dist * ref_fn
            clamped_depth = pt_depth
            if clamped_depth < 0.0:
                clamped_depth = 0.0
            result = _manifold_set(result, result.count, contact_pt, clamped_depth)
            result.count = result.count + 1

    # If clipping produced nothing (degenerate), fall back to single point
    if result.count == 0:
        cp = box_box_contact_point(
            pos_a,
            R_a,
            ha_x,
            ha_y,
            ha_z,
            pos_b,
            R_b,
            hb_x,
            hb_y,
            hb_z,
            normal,
        )
        result.p0 = cp
        result.d0 = depth
        result.count = 1

    return result


# ---------------------------------------------------------------------------
# Box-ground multi-point manifold (vertex enumeration)
# ---------------------------------------------------------------------------


@wp.func
def box_ground_manifold(
    pos: wp.vec3,
    R: wp.mat33,
    hx: float,
    hy: float,
    hz: float,
    ground_z: float,
) -> ContactPolyManifold:
    """Multi-point contact for box vs flat ground (z = ground_z).

    Checks all 8 box vertices against the ground plane.
    Keeps up to 4 deepest penetrating vertices.
    """
    result = ContactPolyManifold()
    result.count = 0

    # All 8 local-frame vertices
    # We test them all but only keep up to 4 (deepest)
    # For a resting box, exactly 4 bottom-face vertices penetrate.
    for vi in range(8):
        sx = hx
        if vi & 1 == 0:
            sx = -hx
        sy = hy
        if vi & 2 == 0:
            sy = -hy
        sz = hz
        if vi & 4 == 0:
            sz = -hz

        local_v = wp.vec3(sx, sy, sz)
        world_v = pos + R * local_v
        vert_depth = ground_z - world_v[2]

        if vert_depth > 0.0:
            cp = wp.vec3(world_v[0], world_v[1], ground_z)
            if result.count < 4:
                result = _manifold_set(result, result.count, cp, vert_depth)
                result.count = result.count + 1
            else:
                # Replace shallowest if this vertex is deeper
                min_d = result.d0
                min_idx = 0
                if result.d1 < min_d:
                    min_d = result.d1
                    min_idx = 1
                if result.d2 < min_d:
                    min_d = result.d2
                    min_idx = 2
                if result.d3 < min_d:
                    min_d = result.d3
                    min_idx = 3
                if vert_depth > min_d:
                    result = _manifold_set(result, min_idx, cp, vert_depth)

    return result


# ---------------------------------------------------------------------------
# ConvexHull support point (GPU linear scan)
# ---------------------------------------------------------------------------

CONVEX_MARGIN = wp.constant(CONTACT_CONVEX_MARGIN)  # see physics.contact_tolerances


@wp.func
def _convexhull_support_local(
    hull_verts: wp.array2d(dtype=wp.float32),
    adr: int,
    count: int,
    direction: wp.vec3,
) -> wp.vec3:
    """Support point of convex hull in local frame (linear scan).

    Returns the vertex with maximum dot product with direction.
    """
    best_dot = float(-1.0e30)
    best_x = float(0.0)
    best_y = float(0.0)
    best_z = float(0.0)
    for i in range(count):
        idx = adr + i
        vx = hull_verts[idx, 0]
        vy = hull_verts[idx, 1]
        vz = hull_verts[idx, 2]
        d = vx * direction[0] + vy * direction[1] + vz * direction[2]
        if d > best_dot:
            best_dot = d
            best_x = vx
            best_y = vy
            best_z = vz
    return wp.vec3(best_x, best_y, best_z)


@wp.func
def _support_world(
    shape_type: int,
    pos: wp.vec3,
    R: wp.mat33,
    params: wp.vec4,
    hull_verts: wp.array2d(dtype=wp.float32),
    hull_adr: int,
    hull_count: int,
    direction: wp.vec3,
) -> wp.vec3:
    """Generic support point in world frame for any shape type.

    Transforms direction to local frame, computes local support, transforms back.
    """
    Rt = wp.transpose(R)
    d_local = Rt * direction

    sup_local = wp.vec3(0.0, 0.0, 0.0)
    if shape_type == SHAPE_SPHERE:
        r = params[0]
        sup_local = r * wp.normalize(d_local)
    elif shape_type == SHAPE_BOX:
        hx = params[0]
        hy = params[1]
        hz = params[2]
        sx = hx
        if d_local[0] < 0.0:
            sx = -hx
        sy = hy
        if d_local[1] < 0.0:
            sy = -hy
        sz = hz
        if d_local[2] < 0.0:
            sz = -hz
        sup_local = wp.vec3(sx, sy, sz)
    elif shape_type == SHAPE_CAPSULE:
        r = params[0]
        hl = params[1]
        axis_dot = d_local[2]
        center_z = hl
        if axis_dot < 0.0:
            center_z = -hl
        d_len = wp.length(d_local)
        if d_len > 1.0e-10:
            sup_local = wp.vec3(0.0, 0.0, center_z) + r * d_local / d_len
        else:
            sup_local = wp.vec3(0.0, 0.0, center_z)
    elif shape_type == SHAPE_CONVEXHULL:
        sup_local = _convexhull_support_local(hull_verts, hull_adr, hull_count, d_local)
    else:
        # Cylinder or unknown: use sphere fallback with mean half-extent
        r = params[0]
        d_len = wp.length(d_local)
        if d_len > 1.0e-10:
            sup_local = r * d_local / d_len

    return pos + R * sup_local


# ---------------------------------------------------------------------------
# GPU GJK closest-distance (convex margin approach, no EPA)
# ---------------------------------------------------------------------------


@wp.struct
class GJKSimplex:
    """GJK simplex with 1-4 points in Minkowski difference space."""

    a: wp.vec3
    b: wp.vec3
    c: wp.vec3
    d: wp.vec3
    n: wp.int32


@wp.func
def _gjk_support(
    pos_a: wp.vec3,
    R_a: wp.mat33,
    type_a: int,
    params_a: wp.vec4,
    pos_b: wp.vec3,
    R_b: wp.mat33,
    type_b: int,
    params_b: wp.vec4,
    hull_verts: wp.array2d(dtype=wp.float32),
    hull_adr_a: int,
    hull_count_a: int,
    hull_adr_b: int,
    hull_count_b: int,
    direction: wp.vec3,
) -> wp.vec3:
    """Minkowski difference support: sup_A(d) - sup_B(-d)."""
    neg_d = wp.vec3(-direction[0], -direction[1], -direction[2])
    sa = _support_world(type_a, pos_a, R_a, params_a, hull_verts, hull_adr_a, hull_count_a, direction)
    sb = _support_world(type_b, pos_b, R_b, params_b, hull_verts, hull_adr_b, hull_count_b, neg_d)
    return sa - sb


@wp.func
def _triple_product(a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.vec3:
    """Vector triple product: (a × b) × c = b(a·c) - a(b·c)."""
    return b * wp.dot(a, c) - a * wp.dot(b, c)


@wp.func
def _gjk_do_simplex_2(s: GJKSimplex, direction: wp.vec3) -> GJKSimplex:
    """Process line simplex (2 points). Returns updated simplex + new direction."""
    ab = s.b - s.a  # newest to oldest: a is newest
    # Swap convention: in our GJK, 'a' is the NEWEST point
    # Actually let's use: s.a = newest, s.b = previous
    ao = wp.vec3(-s.a[0], -s.a[1], -s.a[2])
    if wp.dot(ab, ao) > 0.0:
        # Origin is in the direction of B from A — keep line
        s.n = 2
    else:
        # Origin is behind A — keep only A
        s.b = s.a
        s.n = 1
    return s


@wp.func
def gjk_closest_distance(
    pos_a: wp.vec3,
    R_a: wp.mat33,
    type_a: int,
    params_a: wp.vec4,
    pos_b: wp.vec3,
    R_b: wp.mat33,
    type_b: int,
    params_b: wp.vec4,
    hull_verts: wp.array2d(dtype=wp.float32),
    hull_adr_a: int,
    hull_count_a: int,
    hull_adr_b: int,
    hull_count_b: int,
) -> wp.vec3:
    """GJK closest-distance between two convex shapes.

    Returns vec3(distance, hit, _) where:
      - distance: closest distance between shapes (0 if overlapping)
      - hit: 1.0 if distance < CONVEX_MARGIN (contact detected), else 0.0

    Uses the convex margin approach: contact when distance < margin,
    depth = margin - distance. No EPA needed.

    Reference: Ericson (2004) §4.3, van den Bergen (2003).
    """
    margin = CONVEX_MARGIN

    # Initial direction
    direction = pos_b - pos_a
    if wp.length(direction) < 1.0e-10:
        direction = wp.vec3(1.0, 0.0, 0.0)

    # First support point
    sup = _gjk_support(
        pos_a,
        R_a,
        type_a,
        params_a,
        pos_b,
        R_b,
        type_b,
        params_b,
        hull_verts,
        hull_adr_a,
        hull_count_a,
        hull_adr_b,
        hull_count_b,
        direction,
    )

    # Simplex: track up to 3 points.
    # All variables that are mutated inside the loop must be explicitly typed.
    s_ax = float(sup[0])
    s_ay = float(sup[1])
    s_az = float(sup[2])
    s_bx = float(0.0)
    s_by = float(0.0)
    s_bz = float(0.0)
    s_cx = float(0.0)
    s_cy = float(0.0)
    s_cz = float(0.0)
    s_n = int(1)
    closest_dist = float(wp.length(sup))
    done = int(0)  # 1=converged, 2=overlapping
    dx = float(direction[0])
    dy = float(direction[1])
    dz = float(direction[2])

    for _iter in range(32):
        if done > 0:
            continue  # Warp has no break in static loops; use guard

        s_a = wp.vec3(s_ax, s_ay, s_az)
        s_b = wp.vec3(s_bx, s_by, s_bz)
        s_c = wp.vec3(s_cx, s_cy, s_cz)

        # Direction toward origin from closest simplex feature
        if s_n == 1:
            dx = -s_ax
            dy = -s_ay
            dz = -s_az
        elif s_n == 2:
            ab = s_b - s_a
            ao = wp.vec3(-s_ax, -s_ay, -s_az)
            tp = _triple_product(ab, ao, ab)
            dx = tp[0]
            dy = tp[1]
            dz = tp[2]
        else:
            ab = s_b - s_a
            ac = s_c - s_a
            n = wp.cross(ab, ac)
            ao = wp.vec3(-s_ax, -s_ay, -s_az)
            if wp.dot(n, ao) < 0.0:
                n = wp.vec3(-n[0], -n[1], -n[2])
            dx = n[0]
            dy = n[1]
            dz = n[2]

        direction = wp.vec3(dx, dy, dz)
        d_len = wp.length(direction)
        if d_len < 1.0e-10:
            done = 2  # overlapping
            continue

        direction = direction / d_len

        # New support point
        sup = _gjk_support(
            pos_a,
            R_a,
            type_a,
            params_a,
            pos_b,
            R_b,
            type_b,
            params_b,
            hull_verts,
            hull_adr_a,
            hull_count_a,
            hull_adr_b,
            hull_count_b,
            direction,
        )

        # Check progress
        proj = wp.dot(sup, direction)
        if proj < 1.0e-6:
            done = 1  # converged
            continue

        if proj < closest_dist:
            closest_dist = proj

        # Add to simplex + reduce
        if s_n == 1:
            s_bx = s_ax
            s_by = s_ay
            s_bz = s_az
            s_ax = sup[0]
            s_ay = sup[1]
            s_az = sup[2]
            s_n = 2
            ab2 = wp.vec3(s_bx - s_ax, s_by - s_ay, s_bz - s_az)
            ao2 = wp.vec3(-s_ax, -s_ay, -s_az)
            if wp.dot(ab2, ao2) <= 0.0:
                s_ax = sup[0]
                s_ay = sup[1]
                s_az = sup[2]
                s_n = 1
        elif s_n == 2:
            s_cx = s_bx
            s_cy = s_by
            s_cz = s_bz
            s_bx = s_ax
            s_by = s_ay
            s_bz = s_az
            s_ax = sup[0]
            s_ay = sup[1]
            s_az = sup[2]
            s_n = 3
            # Check which feature is closest
            s_a2 = wp.vec3(s_ax, s_ay, s_az)
            s_b2 = wp.vec3(s_bx, s_by, s_bz)
            s_c2 = wp.vec3(s_cx, s_cy, s_cz)
            ab3 = s_b2 - s_a2
            ac3 = s_c2 - s_a2
            ao3 = wp.vec3(-s_ax, -s_ay, -s_az)
            abc3 = wp.cross(ab3, ac3)
            ab_perp = wp.cross(ab3, abc3)
            ac_perp = wp.cross(abc3, ac3)
            if wp.dot(ab_perp, ao3) > 0.0:
                # Closest to AB edge — drop C
                s_n = 2
            elif wp.dot(ac_perp, ao3) > 0.0:
                # Closest to AC edge — drop B, keep A and C
                s_bx = s_cx
                s_by = s_cy
                s_bz = s_cz
                s_n = 2
            else:
                # Inside triangle — ensure correct winding
                if wp.dot(abc3, ao3) < 0.0:
                    tmpx = s_bx
                    tmpy = s_by
                    tmpz = s_bz
                    s_bx = s_cx
                    s_by = s_cy
                    s_bz = s_cz
                    s_cx = tmpx
                    s_cy = tmpy
                    s_cz = tmpz
        else:
            # s_n == 3: replace farthest point
            s_a3 = wp.vec3(s_ax, s_ay, s_az)
            s_b3 = wp.vec3(s_bx, s_by, s_bz)
            s_c3 = wp.vec3(s_cx, s_cy, s_cz)
            da = wp.dot(s_a3, s_a3)
            db = wp.dot(s_b3, s_b3)
            dc = wp.dot(s_c3, s_c3)
            if da >= db and da >= dc:
                s_ax = sup[0]
                s_ay = sup[1]
                s_az = sup[2]
            elif db >= dc:
                s_bx = sup[0]
                s_by = sup[1]
                s_bz = sup[2]
            else:
                s_cx = sup[0]
                s_cy = sup[1]
                s_cz = sup[2]

    # Compute final distance
    # NOTE: this simplified GJK uses a 3-point (triangle) simplex max.
    # In 3D, overlap detection requires a 4-point tetrahedron. If the loop
    # didn't converge to separation (done != 1), assume overlap (conservative).
    dist = float(0.0)
    if done == 1:
        # Converged to separation — use tracked closest distance
        if s_n == 1:
            dist = wp.length(wp.vec3(s_ax, s_ay, s_az))
        else:
            dist = wp.max(closest_dist, 0.0)
    else:
        # done==0 (no convergence) or done==2 (explicit overlap) → overlap
        dist = 0.0

    hit = float(0.0)
    if dist < margin:
        hit = 1.0

    return wp.vec3(dist, hit, 0.0)


@wp.func
def gjk_contact_normal(
    pos_a: wp.vec3,
    R_a: wp.mat33,
    type_a: int,
    params_a: wp.vec4,
    pos_b: wp.vec3,
    R_b: wp.mat33,
    type_b: int,
    params_b: wp.vec4,
    hull_verts: wp.array2d(dtype=wp.float32),
    hull_adr_a: int,
    hull_count_a: int,
    hull_adr_b: int,
    hull_count_b: int,
) -> wp.vec3:
    """Contact normal from B to A using support point sampling."""
    d = pos_a - pos_b
    d_len = wp.length(d)
    if d_len > 1.0e-10:
        return d / d_len
    return wp.vec3(0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# ConvexHull-ground multi-point manifold (vertex enumeration)
# ---------------------------------------------------------------------------


@wp.func
def convexhull_ground_manifold(
    pos: wp.vec3,
    R: wp.mat33,
    hull_verts: wp.array2d(dtype=wp.float32),
    adr: int,
    count: int,
    ground_z: float,
) -> ContactPolyManifold:
    """Multi-point ground contact for ConvexHull via vertex enumeration.

    Checks all hull vertices against ground plane, keeps up to 4 deepest.
    """
    result = ContactPolyManifold()
    result.count = 0

    for i in range(count):
        idx = adr + i
        local_v = wp.vec3(hull_verts[idx, 0], hull_verts[idx, 1], hull_verts[idx, 2])
        world_v = pos + R * local_v
        vert_depth = ground_z - world_v[2]

        if vert_depth > 0.0:
            cp = wp.vec3(world_v[0], world_v[1], ground_z)
            if result.count < 4:
                result = _manifold_set(result, result.count, cp, vert_depth)
                result.count = result.count + 1
            else:
                # Replace shallowest
                min_d = result.d0
                min_idx = 0
                if result.d1 < min_d:
                    min_d = result.d1
                    min_idx = 1
                if result.d2 < min_d:
                    min_d = result.d2
                    min_idx = 2
                if result.d3 < min_d:
                    min_d = result.d3
                    min_idx = 3
                if vert_depth > min_d:
                    result = _manifold_set(result, min_idx, cp, vert_depth)

    return result
