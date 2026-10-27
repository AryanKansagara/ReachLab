# ReachLab

A state-based embodied-agent experiment: a PyTorch policy learns damped-Jacobian reaching behavior for a seven-joint KUKA iiwa in PyBullet DIRECT simulation.

## Run

Python 3.12; CPU only. Install the pinned dependencies into a virtual environment:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest -v
.venv/bin/python reachlab.py --output artifacts
```

PyBullet 3.2.7 may require a source build on Apple Silicon. With Apple Clang 21, its bundled zlib has a header-order conflict with `fdopen`; installing with `CFLAGS='-include stdio.h'` works around the conflicting declaration. Linux wheels may avoid the source build.

## Experiment design

- PyBullet loads the bundled KUKA iiwa URDF, uses a fixed base, gravity and joint-position motor control. Each control update advances 24 physics steps at 240 Hz.
- An analytic damped-least-squares Jacobian controller supplies labels. A 10-dimensional observation contains seven joint offsets and three target-error coordinates. Actions are seven bounded joint-position increments.
- A two-hidden-layer neural policy is trained with supervised behavioral cloning, MSE, AdamW weight decay and dropout. This is **not reinforcement learning**, a foundation model or an LLM agent.
- Independent fixed seeds generate 6,000 training and 1,000 validation state/error samples. Checkpoint selection uses validation loss only. Thirty independently seeded reachable goals are evaluated after training with a 2.5 cm threshold and 50-step horizon; targets within 4 cm of the initial position are excluded.
- The learned policy, analytic expert and zero-action baseline receive the same evaluation goals. Evaluation starts at the same home configuration. This is a narrow-workspace generalization check, not held-out robot/task or broad out-of-distribution testing.

`artifacts/` holds generated policy weights and metrics and is ignored by Git. `results/metrics.json` records the observed local run without distributing third-party assets. Numerical outcomes are reported rather than enforced as success guarantees.

## Tests and boundaries

Fifteen tests cover finite-value and shape validation, action bounds, singular Jacobians, observation construction, deterministic evaluation, simulator reset/forward kinematics, joint limits, expert reaching, seeded sampling, split separation and checkpoint round-tripping.

Only joint/target state is observed; there are no camera inputs, grasping tasks, collision-avoidance planning, language commands, Isaac Sim, ManiSkill, hardware deployment or sim-to-real results. Joint limits and action clipping are numerical guardrails, not a physical safety certification. No internal Huawei product integration is claimed. Code was developed with AI assistance; inspect it and describe your contribution accurately.

See [research report](docs/RESEARCH_REPORT.md) for assumptions, results and next steps, and the official [PyBullet guide](https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstart_guide/PyBulletQuickstartGuide.md.html) for simulator APIs. The workflow in `ci/github-actions.yml` is an inactive template; move it into `.github/workflows` to activate it with appropriate permissions.
