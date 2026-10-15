"""Scalar prediction and actual H200 first-contact measurements."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

source = Path("/tmp/admm_gpu_iteration_probe.log")
rows = [json.loads(line) for line in source.read_text().splitlines() if line.startswith('{"mass"')]
fig, ax = plt.subplots(figsize=(8.5, 4.5))
iterations = np.arange(1, 501)
for mass in [.1, 1., 50., 1000.]:
    points = [r for r in rows if r["mass"] == mass]
    q = points[0]["W"] + points[0]["R"]
    curve = 100*(1/(1+q))**iterations
    line, = ax.semilogy(iterations, curve, label=f"{mass:g} kg: scalar prediction")
    ax.scatter([r["iterations"] for r in points], [100*r["relative_error"] for r in points],
               color=line.get_color(), marker="o", zorder=5)
ax.axvline(30, color="gray", linestyle=":", linewidth=1)
ax.text(34, .2, "GPU default: 30", fontsize=9, color="gray")
ax.set(xlabel="ADMM iterations", ylabel="Relative normal impulse error (%)",
       title="First contact, fixed rho=1: mass scaling controls convergence",
       xlim=(0, 500), ylim=(1e-7, 110))
ax.grid(alpha=.25)
ax.legend(fontsize=8, loc="center right")
fig.text(.5, .015, "Lines: exact scalar recurrence using measured W+R. Dots: fresh H200 results at 30/300 iterations.",
         ha="center", fontsize=8)
fig.tight_layout(rect=(0,.045,1,1))
fig.savefig("/tmp/admm_convergence_mass_scale.png", dpi=160)
