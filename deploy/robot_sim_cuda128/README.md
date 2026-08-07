# robot_sim CUDA 12.8 validation environment

This directory records the environment used to build and validate the CUDA
optical extensions on `192.168.80.101`. It contains manifests only; wheel
binaries and environment directories are intentionally not committed.

## Recorded runtime

- Conda environment name: `robot_sim`
- Python: 3.12.13
- PyTorch: 2.9.1+cu128
- CUDA toolkit/NVCC: 12.8
- Warp: 1.12.0
- pytest: 9.1.1
- ruff: 0.16.1

The environment path used during validation was
`/home/ang.gao/anaconda3/envs/robot_sim`. That path is machine-specific and is
not part of the runtime contract.

## Manifest files

- `conda-explicit.txt`: output of `conda list -n robot_sim --explicit`.
- `python-packages.txt`: dependency inventory from
  `conda run -n robot_sim python -m pip freeze --exclude-editable`.
- `wheelhouse.sha256`: SHA-256 hashes for the 62 wheels relayed from the
  workstation to the server.

Generate or refresh the inventories on the target machine:

```bash
mkdir -p deploy/robot_sim_cuda128
conda list -n robot_sim --explicit > deploy/robot_sim_cuda128/conda-explicit.txt
conda run -n robot_sim python -m pip freeze --exclude-editable \
  > deploy/robot_sim_cuda128/python-packages.txt
(cd /data3/ang.gao/.package-relay/robot_sim_20260806 && \
  sha256sum -- *.whl | LC_ALL=C sort) > deploy/robot_sim_cuda128/wheelhouse.sha256
```

Verify the server wheelhouse against the tracked manifest:

```bash
cd /data3/ang.gao/.package-relay/robot_sim_20260806
sha256sum --check /data3/ang.gao/robot_simulator/deploy/robot_sim_cuda128/wheelhouse.sha256
```

When direct package-network access is unavailable, transfer the wheelhouse from
the workstation and verify this manifest before installation. Do not substitute
different PyTorch, CUDA, Warp, or transitive package versions merely because a
remote index times out.

Example offline Python-package installation after verification:

```bash
conda run -n robot_sim python -m pip install \
  --no-index \
  --find-links /data3/ang.gao/.package-relay/robot_sim_20260806 \
  torch==2.9.1+cu128 \
  warp-lang==1.12.0 \
  numpy==2.5.1 \
  pytest==9.1.1 \
  ruff==0.16.1

conda run -n robot_sim python -m pip install --no-deps -e .
```

The wheelhouse contains the reviewed transitive dependencies for these exact
pins. `python-packages.txt` is an evidence inventory, not a requirements file;
it includes environment tooling and deliberately excludes the editable project
checkout so the manifest does not contain a circular Git revision reference.

## CUDA extension execution

The environment stores these variables:

```text
CUDA_HOME=/home/ang.gao/anaconda3/envs/robot_sim
TORCH_EXTENSIONS_DIR=/data3/ang.gao/.cache/torch_extensions_robot_sim_cuda128
```

Representative validation command:

```bash
conda run -n robot_sim env \
  PYTHONPATH=. \
  CUDA_VISIBLE_DEVICES=0 \
  python -m pytest tests/gpu/test_optical_gpu_runtime.py \
  -k "cuda and (camera or raygen)" -q
```

For the multi-GPU device-guard test, expose two devices explicitly:

```bash
conda run -n robot_sim env \
  PYTHONPATH=. \
  CUDA_VISIBLE_DEVICES=0,1 \
  python -m pytest tests/gpu/test_optical_gpu_runtime.py \
  -k "guard_target_device_and_restore_ambient_device" -q
```
