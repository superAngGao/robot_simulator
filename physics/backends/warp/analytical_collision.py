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

# Shape type constants (must match static_data.py)
SHAPE_NONE = wp.constant(0)
SHAPE_SPHERE = wp.constant(1)
SHAPE_BOX = wp.constant(2)
SHAPE_CYLINDER = wp.constant(3)
SHAPE_CAPSULE = wp.constant(4)


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
