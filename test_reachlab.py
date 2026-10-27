import unittest

import numpy as np
import torch

from reachlab import Arm, HOME, MAX_DELTA, Policy, dataset, expert_action, finite_vector, observations, train


class MathTests(unittest.TestCase):
    def test_bad_vector(self):
        for value in ([1, 2], [np.nan, 0, 0], [np.inf, 0, 0]):
            with self.assertRaises(ValueError):
                finite_vector(value, 3)

    def test_observation_shape(self):
        obs = observations(HOME, np.zeros(3))
        self.assertEqual(obs.shape, (10,))
        self.assertTrue(np.all(obs == 0))

    def test_expert_zero(self):
        np.testing.assert_array_equal(expert_action(np.ones((3, 7)), np.zeros(3)), np.zeros(7))

    def test_singular_jacobian(self):
        np.testing.assert_array_equal(expert_action(np.zeros((3, 7)), np.ones(3)), np.zeros(7))

    def test_action_bound(self):
        action = expert_action(np.eye(3, 7), np.ones(3) * 100)
        self.assertLessEqual(np.max(np.abs(action)), MAX_DELTA)

    def test_bad_jacobian(self):
        for value in (np.zeros((7, 3)), np.full((3, 7), np.nan)):
            with self.assertRaises(ValueError):
                expert_action(value, np.ones(3))

    def test_policy_shape_and_bound(self):
        result = Policy()(torch.randn(5, 10))
        self.assertEqual(tuple(result.shape), (5, 7))
        self.assertTrue(torch.all(result.abs() <= MAX_DELTA))

    def test_evaluation_deterministic(self):
        model = Policy().eval()
        obs = torch.zeros(10)
        torch.testing.assert_close(model(obs), model(obs))


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.arm = Arm()

    def tearDown(self):
        self.arm.close()

    def test_reset_and_fk(self):
        np.testing.assert_allclose(self.arm.joints(), HOME)
        self.assertTrue(np.isfinite(self.arm.position()).all())

    def test_joint_limits(self):
        with self.assertRaises(ValueError):
            self.arm.reset(np.ones(7) * 100)

    def test_jacobian_shape(self):
        self.assertEqual(self.arm.jacobian().shape, (3, 7))

    def test_expert_reduces_reaching_error(self):
        goal = self.arm.position() + np.array([0.02, 0.02, -0.02])
        before = np.linalg.norm(goal - self.arm.position())
        for _ in range(10):
            self.arm.step(expert_action(self.arm.jacobian(), goal - self.arm.position()))
        self.assertLess(np.linalg.norm(goal - self.arm.position()), before)

    def test_dataset_seeded(self):
        x, y = dataset(self.arm, 12, 10)
        xx, yy = dataset(self.arm, 12, 10)
        torch.testing.assert_close(x, xx)
        torch.testing.assert_close(y, yy)

    def test_dataset_splits_differ(self):
        x, _ = dataset(self.arm, 12, 10)
        xx, _ = dataset(self.arm, 12, 11)
        self.assertFalse(torch.equal(x, xx))

    def test_training_and_checkpoint_roundtrip(self):
        torch.set_num_threads(1)
        x, y = dataset(self.arm, 32, 10)
        vx, vy = dataset(self.arm, 12, 11)
        model, stats = train(x, y, vx, vy, epochs=3)
        self.assertTrue(np.isfinite(stats['validation_mse_rad2']))
        restored = Policy().eval()
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(model(vx), restored(vx))


if __name__ == '__main__':
    unittest.main()
