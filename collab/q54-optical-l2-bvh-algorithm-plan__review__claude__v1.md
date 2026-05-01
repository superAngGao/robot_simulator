Initiative: q54-optical-l2-bvh-algorithm
Stage: review
Author: claude
Version: v1
Date: 2026-05-01
Status: accepted-with-required-edits
Related Files: collab/q54-optical-l2-bvh-algorithm-plan__codex__v1.md, optics/scene.py, optics/execution.py
Owner Summary: Claude accepted the overall L2 BVH path but required edits around source-order tie-break semantics, world-space rebuild tradeoffs, missing-acceleration error typing, tie-break epsilon, node container wording, and inclusive ray-AABB boundary behavior.

# Q54 L2 BVH Algorithm Plan Review

## Conclusion

The algorithm plan is clear and directionally correct. The proposed first
implementation can remain:

```text
in-repo CPU BVH
world-space packed triangles
longest-axis median split
leaf_size = 4
iterative traversal
analytical plane side pass
scene/cache build, executor consume
```

However, several edits are required before implementation.

## Required Edits

### 1. Source Order Must Be Global

The most important issue is plane source order.

The algorithm plan said plane source order could come after triangle primitive
source order. That breaks reference parity. If a plane instance appears before a
mesh instance in `snapshot.instances`, the reference executor scans the plane
first. A BVH executor must preserve that tie-break behavior.

Correct rule:

```text
source order is global over snapshot.instances
geometry kind must not change source order
```

Use a lexicographic key:

```text
(instance_index, primitive_index_within_instance)
```

For a plane:

```text
(instance_index, 0)
```

For mesh triangle `j`:

```text
(instance_index, j)
```

Tie-break should compare this key, not a triangles-only integer order.

### 2. World-Space Rebuild Tradeoff

World-space packed triangles are acceptable for L2 correctness-first work, but
the document must state the cost explicitly:

```text
every BVH snapshot rebuild transforms all mesh vertices
```

Future triggers for local-space triangles plus per-instance transforms:

```text
multi-env batching
many rigid instances sharing mesh topology
deformable mesh producer
BVH refit support
Embree/OptiX adapter needs
```

### 3. Missing Acceleration Error

Raising on missing acceleration is correct, but use a dedicated exception class
instead of a generic `ValueError`.

Suggested name:

```text
MissingAccelerationError
```

This lets callers distinguish configuration errors from intersection failures.

### 4. Tie-Break Epsilon

Do not reuse build epsilon for hit-distance tie-breaks. Define a separate value:

```text
t_eps = 1e-9
```

Add tests that cover equal-distance / coplanar behavior so this tolerance does
not accidentally hide real nearest-hit differences.

### 5. Node Container Wording

Use `list[CpuBvhNode]` in the first implementation. It is the natural build-time
container and still supports O(1) indexed access during traversal. A compact
array representation can come later after profiling.

### 6. Ray-AABB Inclusive Boundaries

The slab test should treat boundary hits inclusively:

```text
origin on slab boundary is inside
t_exit >= max(t_enter, 0.0)
```

This matches the inclusive triangle semantics:

```text
u >= 0
v >= 0
u + v <= 1
```

## Review Question Answers

1. World-space packed triangles are acceptable for L2, with rebuild cost
   documented.
2. Median split is sufficient; defer SAH.
3. `leaf_size = 4` is reasonable.
4. Skipping degenerate triangles matches reference behavior.
5. Source-order tie-break is sufficient once `t_eps` is defined separately.
6. Plane source order must follow global snapshot instance order, not "after
   triangles".
7. Raise on missing acceleration; use a dedicated exception class.
8. `snapshot_from_frame_inputs(..., acceleration="cpu_bvh")` is the right opt-in
   API.
