"""State-based imitation learning for simulated KUKA reaching, not real hardware."""
import argparse
import json
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data
import torch
from torch import nn

HOME = np.array([0.0, 0.5, 0.0, -1.2, 0.0, 1.0, 0.0])
MAX_DELTA = 0.12


def finite_vector(value, size):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f"expected finite vector of length {size}")
    return value


def expert_action(jacobian, error):
    jacobian = np.asarray(jacobian, dtype=np.float64)
    if jacobian.shape != (3, 7) or not np.isfinite(jacobian).all():
        raise ValueError("expected finite 3x7 Jacobian")
    error = finite_vector(error, 3)
    # Damped least squares avoids unstable updates near singular configurations.
    delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 0.01 * np.eye(3), error)
    return np.clip(delta, -MAX_DELTA, MAX_DELTA)


class Arm:
    def __init__(self):
        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(1 / 240, physicsClientId=self.client)
        self.robot = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True, physicsClientId=self.client)
        info = [p.getJointInfo(self.robot, i, physicsClientId=self.client) for i in range(7)]
        self.lower = np.array([item[8] for item in info])
        self.upper = np.array([item[9] for item in info])
        self.reset(HOME)

    def close(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)

    def reset(self, q):
        q = finite_vector(q, 7)
        if np.any(q < self.lower) or np.any(q > self.upper):
            raise ValueError("joint position outside limits")
        for i, value in enumerate(q):
            p.resetJointState(self.robot, i, float(value), physicsClientId=self.client)

    def joints(self):
        return np.array([item[0] for item in p.getJointStates(self.robot, list(range(7)), physicsClientId=self.client)])

    def position(self):
        return np.array(p.getLinkState(self.robot, 6, computeForwardKinematics=True, physicsClientId=self.client)[4])

    def jacobian(self):
        q = self.joints().tolist()
        linear, _ = p.calculateJacobian(self.robot, 6, [0, 0, 0], q, [0.] * 7, [0.] * 7, physicsClientId=self.client)
        return np.array(linear)

    def step(self, delta):
        delta = np.clip(finite_vector(delta, 7), -MAX_DELTA, MAX_DELTA)
        target = np.clip(self.joints() + delta, self.lower, self.upper)
        p.setJointMotorControlArray(self.robot, list(range(7)), p.POSITION_CONTROL,
                                   targetPositions=target.tolist(), forces=[200.] * 7,
                                   physicsClientId=self.client)
        for _ in range(24):
            p.stepSimulation(physicsClientId=self.client)
        return self.position()


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(10, 128), nn.ReLU(), nn.Dropout(0.05),
                                    nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 7), nn.Tanh())

    def forward(self, obs):
        return self.layers(obs) * MAX_DELTA


def observations(q, error):
    return np.concatenate(((finite_vector(q, 7) - HOME) / 0.35, finite_vector(error, 3) / 0.1)).astype(np.float32)


def dataset(arm, count, seed):
    rng = np.random.default_rng(seed)
    x, y = [], []
    for _ in range(count):
        q = HOME + rng.uniform(-0.35, 0.35, 7)
        arm.reset(q)
        error = rng.uniform(-0.1, 0.1, 3)
        x.append(observations(q, error))
        y.append(expert_action(arm.jacobian(), error))
    return torch.from_numpy(np.array(x)), torch.from_numpy(np.array(y, dtype=np.float32))


def train(x, y, vx, vy, epochs=120):
    torch.manual_seed(7)
    model = Policy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.001)
    best, selected, state = float("inf"), 0, None
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x))
        for batch in order.split(256):
            loss = nn.functional.mse_loss(model(x[batch]), y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation = nn.functional.mse_loss(model(vx), vy).item()
        if validation < best:
            best, selected = validation, epoch + 1
            state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(state)
    model.eval()
    return model, {"validation_mse_rad2": best, "selected_epoch": selected, "epochs": epochs}


def rollouts(arm, model, count=30, seed=303):
    rng = np.random.default_rng(seed)
    goals = []
    arm.reset(HOME)
    initial = arm.position()
    # Do not count targets already inside the success radius as solved tasks.
    while len(goals) < count:
        arm.reset(HOME + rng.uniform(-0.22, 0.22, 7))
        candidate = arm.position()
        if np.linalg.norm(candidate - initial) > 0.04:
            goals.append(candidate)
    results = {}
    for mode in ("learned", "expert", "zero_action"):
        distances = []
        for goal in goals:
            arm.reset(HOME)
            for _ in range(50):
                error = goal - arm.position()
                if np.linalg.norm(error) <= 0.025:
                    break
                if mode == "expert":
                    delta = expert_action(arm.jacobian(), error)
                elif mode == "zero_action":
                    delta = np.zeros(7)
                else:
                    with torch.inference_mode():
                        delta = model(torch.from_numpy(observations(arm.joints(), error))).numpy()
                arm.step(delta)
            distances.append(float(np.linalg.norm(goal - arm.position())))
        results[mode] = {"successes": sum(d <= 0.025 for d in distances), "episodes": count,
                         "mean_final_distance_m": float(np.mean(distances)), "final_distances_m": distances}
    return results


def experiment(output):
    torch.set_num_threads(1)
    arm = Arm()
    try:
        x, y = dataset(arm, 6000, 101)
        vx, vy = dataset(arm, 1000, 202)
        model, training = train(x, y, vx, vy)
        metrics = {"environment": "PyBullet DIRECT, KUKA iiwa, CPU simulation only",
                   "method": "state-based behavioral cloning of damped-Jacobian expert; not reinforcement learning",
                   "train_samples": 6000, "validation_samples": 1000,
                   "training_seed": 101, "validation_seed": 202, "evaluation_seed": 303,
                   "success_threshold_m": 0.025, "max_control_steps": 50,
                   "training": training, "evaluation": rollouts(arm, model),
                   "limitations": ["No hardware, grasping, visual inputs, LLM or collision-avoidance planning.",
                                   "Independent seeded samples/goals from one narrow workspace, not broad out-of-distribution evaluation.",
                                   "Expert labels and learned policy use privileged joint/target state."]}
        output.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output / "policy.pt")
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics, indent=2))
    finally:
        arm.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    experiment(args.output)
