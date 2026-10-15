# Q54 P12.4 C3b prebound Warp scene/refit implementation note v2

Status: follow-up implementation for Claude re-review. This note supplements
v1 and addresses both P1 findings on commit `130e646`. No benchmark result or
baseline is produced.

## Factories are now immutable and self-auditing

Both `WarpSceneUpdateCommandFactory` and `WarpBvhRefitCommandFactory` now seal
their configuration at the end of construction and reject every later normal
attribute assignment.

Construction also captures private immutable snapshots of the facts needed to
prove the factory has not drifted:

- indexed CUDA device and nonzero prepared stream handle;
- native-input manifest and input-contract provenance;
- scene-contract digest;
- exact local-scene or immutable-BVH storage manifest;
- for refit, native-input topology digest, actual BVH topology digest,
  topology-origin ID, and normalized topology facts.

Every `seal()` starts by recomputing and comparing those facts. Therefore even
a test that deliberately bypasses the read-only API with
`object.__setattr__()` cannot turn a changed manifest, stream, scene, topology,
or storage graph into valid evidence. Validation occurs before `wp.launch(...,
record_cmd=True)` records the first command.

The target C4 bounds slot remains a separate `seal(..., bvh=slot_bvh)` input.
It must share the factory's exact immutable topology and primitive-metadata
allocations; only its mutable node-bound allocations may differ.

## Frozen leaf-4 topology identity

The formal topology identity is now explicit:

- acceleration ID: `cpu_median_leaf4_topology_warp_refit_v1`;
- topology origin: `cpu_median_longest_axis_leaf4_source_order_v1`;
- leaf size: `4`;
- split strategy: stable `sort`;
- operation: `warp_bvh_leaf_internal_refit_out_v1`.

`DeviceOpticalBvh` records the topology-origin ID. The generic CPU builder sets
that identity only for leaf-4 + `sort`; other supported compatibility builds
carry no canonical refit origin. `WarpBvhRefitCommandFactory` requires the
formal origin plus the exact leaf-size and split-strategy facts, so a real
leaf-1, leaf-2, leaf-8, or `partition` BVH cannot be sealed under the frozen
acceleration identity.

The topology schema is upgraded to v2. Its digest now binds the builder ID,
actual leaf size, actual split strategy, and every normalized topology array.
The refit-command evidence schema is upgraded to v2 and records both the
acceleration ID and topology-origin ID alongside the actual topology digest.

## Added negative coverage

Real-GPU tests now prove:

- direct assignment to either factory is rejected;
- bypassed manifest mutation is caught by the construction snapshot;
- bypassed stream replacement is caught before recording;
- bypassed refit-topology replacement with a different scene contract is
  caught instead of degenerating into a self-comparison;
- real leaf sizes 1, 2, and 8 and a real leaf-4 `partition` build are all
  rejected by the prepared refit factory;
- the accepted bundle reports the frozen acceleration and topology-origin IDs.

CPU tests prove that topology digests differ when leaf size or split strategy
differs, including cases where array content alone might otherwise be
insufficient to distinguish the build contract.

## Verification

- targeted Ruff checks: passed;
- C3b unit and real-GPU tests: 14 passed;
- benchmark/optics/optical-GPU regression: 450 passed;
- full unit suite: 1139 passed, 20 skipped (one existing pytest deprecation
  warning).

The C4/B3 activation object remains unchanged and false. a3b result emission,
baseline generation, and performance inspection remain closed.
