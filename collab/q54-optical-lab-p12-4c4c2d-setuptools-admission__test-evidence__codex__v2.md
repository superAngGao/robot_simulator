# Target Python CPU helper and record regression

Initiative: Q54 / P12.4 / C4c2 / CUDA G1
Stage: CPU implementation evidence, not execution preflight or acceptance
Author: codex
Version: v2
Date: 2026-10-09
Status: draft
Related Files: setuptools-admission implementation-note v3/v4; design-request/design-review v2
Owner Summary: At exact code553038f, the target Python3.12 suite passed190 CPU tests, including all11 historical archived-input codec tests skipped on Windows. No actual setuptools/backend/compiler was imported or run. Full installed closure, source integration and independent execution preflights remain required.

## Exact candidate and artifacts

- Code:553038f11c24716d23be279dc08ac9e9257efc27.
- Primary host:tileops-dev,183.166.183.33:40031. No fallback was used.
- Isolated tree:/data2/shared/ang.gao/robot_sim_cuda_compilation_wip. Before
  transfer it was clean detached52fb36a8960596ddb2ac3decd8d129188e4b1eb4. A verified
  incremental bundle was fetched, then the clean tree checked out detached553038f.
  No reset, merge, original-tree modification or process/GPU reset occurred.
- Bundle:/data2/shared/ang.gao/setuptools-553038f-wip-v1.bundle,103016bytes,
  SHA25663ae452a5e92491b997b54b44ca2d11f49b15324166a1b4e3f28ba13bc4816dd.
  Its prerequisite is52fb36a; bundle verification succeeded on the target.
- Python:/home/ang.gao/miniforge3/envs/robot_sim/bin/python,
  3.12.13 packaged by Anaconda, Inc., GCC14.3.0.
- Log:/data2/shared/ang.gao/setuptools-553038f-cpu-v1.log,329bytes,
  SHA256b06cd2db63dc090bfc24a19e045a996411f1cdf2d33fda26dd44155f4041c308.
- Historical read-only input:/data2/shared/ang.gao/c2d3-fc86182-native-scope-v2.tar.gz,
  SHA256d872f90f699fea8a1ea3b6a3aecc2c6a481bf34d4936b9a4de732cbaaef23130.
  The codec reads the archive without extraction or modification; the generated
  compilation continuation is expressly synthetic, not a rerun of native inputs.
- Bundle and log also reside in C:\Users\AG\新建文件夹\c4_assets.

## Invocation and result

From the exact clean remote tree, set PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 and
ROBOT_SIM_COMPILATION_INPUT_ARCHIVE to the archive above. Run the interpreter with
`-I -S -B -`, feeding this CPU-only launcher on stdin (stdout/stderr tee'd to the
new log under Bash set -euo pipefail):

```python
import os
import sys
from pathlib import Path
repo = '/data2/shared/ang.gao/robot_sim_cuda_compilation_wip'
site = '/home/ang.gao/miniforge3/envs/robot_sim/lib/python3.12/site-packages'
reserved = {'torch', 'warp', 'setuptools', '_distutils_hack', 'distutils'}
assert sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode
assert not any(name.split('.')[0] in reserved for name in sys.modules)
sys.path[:0] = [repo, site]
import pytest
print('owned_cpu_pid=' + str(os.getpid()), flush=True)
Path('/data2/shared/ang.gao/setuptools-553038f-cpu-v1.pid').write_text(str(os.getpid()))
files = ['test_runtime_environment.py', 'test_setuptools_contract.py', 'test_setuptools_admission.py',
         'test_setuptools_replay.py', 'test_runtime_custody.py', 'test_cuda_compilation_custody.py',
         'test_cuda_toolchain_environment.py', 'test_cuda_compilation_unwind.py', 'test_cuda_compilation_codec.py']
status = pytest.main(['-q', '-p', 'no:cacheprovider'] + ['tests/unit/optics/' + name for name in files])
unexpected = sorted(name for name in sys.modules if name.split('.')[0] in reserved)
print('unexpected_dependency_modules=' + repr(unexpected), flush=True)
assert not unexpected
print('pytest_exit=' + str(status), flush=True)
raise SystemExit(status)
```

The explicit site directory supplies the test framework without executing site
startup or .pth shims. It is a test launcher, not the captured worker's import
topology or a deployment admission mechanism.

Observed:190passed in15.34s; unexpected_dependency_modules=[]; pytest_exit=0.
Owned PID290532 was absent at command completion and on a subsequent explicit
/proc/290532 check. Git status remained clean. No worker, compiler, native
observer, supervisor, subreaper or GPU job was launched by this suite.

## Scope

This adds actual target-interpreter evidence for the CPU helpers and descriptive
parser regressions, including the full historical v1 record codec from an archived
input. New initialization action tests exercise production dispatch, ordinary
file/call joins and guardian adoptions with the outer context supplied descriptively.
They do not validate a complete real v2 compilation record, actual pinned setuptools
initialization, same-object alias integration, worker cwd, native scopes or process
cleanup. The detailed stand-in limits in implementation-note v3/v4 still apply.

No independent preflight/review verdict is claimed. Next freeze the full installed
v6 profile and source/catalog policy, complete the real dependency CPU candidate
and obtain its exact preflight before execution. The original compilation/native
proof, whole checkpoint review, later C4 phases and systematic refactor remain.
