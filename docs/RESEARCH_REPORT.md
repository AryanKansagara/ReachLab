# Research report: state-based reaching

## Question and method

Can a small neural policy imitate a local analytic controller well enough to complete reaching rollouts, rather than merely fit one-step labels? This bounded experiment studies the train/evaluation distinction in a simulator, not general-purpose robotic intelligence.

The analytic controller computes a damped least-squares joint update from a translational Jacobian. The PyTorch policy receives normalized joint offsets and Cartesian target error. Its 10-128-128-7 MLP uses ReLU, 5% dropout and a tanh-bounded output. Training uses MSE and AdamW (learning rate 0.002, weight decay 0.001), 256-sample batches and 120 epochs. The checkpoint with lowest validation MSE is restored before evaluation.

Training and validation use independent seeds (101/202) and 6,000/1,000 random state/error pairs. The rollout goals use seed 303, a different distribution of reachable targets derived from forward kinematics near the home configuration. Goals within 4 cm of the initial end-effector are excluded, so none begins inside the 2.5 cm success region. This is a local-workspace check, not rigorous task-level or robot-level distribution shift.

## Observed local run

Executed on macOS ARM64 CPU with PyTorch 2.10.0 and PyBullet 3.2.7. Source: `results/metrics.json`.

| Controller | Goals reached within 2.5 cm / 30 | Mean final distance |
| --- | ---: | ---: |
| Learned behavioral-cloning policy | 30 | 1.053 cm |
| Analytic damped-Jacobian expert | 30 | 1.089 cm |
| Zero-action baseline | 0 | 11.435 cm |

Selected epoch: 114; validation action MSE: 0.0000128975 rad². All 15 tests passed. All controllers share the same goals, initial state, 50-step horizon and stopping rule. The small mean-distance difference does not establish that the learned policy is superior to its expert: success-triggered stopping, one seed and only 30 episodes preclude that conclusion. Zero action is deliberately weak; a random-policy or tuned feedback baseline would add context.

## Interpretation and limitations

The experiment confirms executable training-to-rollout integration and local imitation of the selected controller. Labels and policy use privileged simulator state. Regularization is implemented but no ablation establishes its causal benefit. A stronger study would repeat training and evaluation seeds, report confidence intervals and action/trajectory efficiency, then test workspace shift, observation noise, perturbations and held-out arm configurations.

There is no perception, natural-language grounding, learned dynamics model, RL reward optimization, grasping, obstacle planning, hardware deployment or internal-product integration. Joint/action bounds are not safety guarantees. The model is small and the workspace intentionally restricted; the result must not be presented as a foundation-model or general robotics benchmark.

Next milestones: add observation noise and multi-seed evaluation; compare behavioral cloning with DAgger or a documented RL baseline; inspect camera-conditioned policies; only then investigate real robot integration with appropriate safety supervision. These are proposals, not implemented features.
