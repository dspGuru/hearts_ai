"""
Additional tests to augment code coverage for hearts_learn.py.
Focuses on edge cases and uncovered code paths.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
import io
import sys

from hearts_game import GameMode, ThreePlayerMode, Card, Suit, Rank, HeartsGame
from hearts_ai import AIWeights, DEFAULT_WEIGHTS, HeartsAI
from hearts_learn import (
    simulate_game,
    compute_reward,
    LearnableWeights,
    HeartsTrainer,
    TrainingStats,
    run_learning_mode,
)


# =============================================================================
# LearnableWeights Extended Tests
# =============================================================================


class TestLearnableWeightsExtended:
    """Extended tests for LearnableWeights class."""

    def test_get_weight_dict(self):
        """Cover get_weight_dict method."""
        initial = AIWeights()
        lw = LearnableWeights(initial)

        weight_dict = lw.get_weight_dict()

        assert isinstance(weight_dict, dict)
        assert "pass_queen_of_spades" in weight_dict
        assert "high_heart_threshold" in weight_dict
        # Integer fields should be integers
        assert isinstance(weight_dict["high_heart_threshold"], int)

    def test_learnable_weights_normalization(self):
        """Cover weight normalization and denormalization."""
        # Test with extreme values
        initial = AIWeights(
            pass_queen_of_spades=50.0,  # At minimum
            moon_threat_threshold=15,  # At maximum
        )
        lw = LearnableWeights(initial)

        # Get weights and verify bounds
        tensor = lw.get_weights_tensor()
        assert tensor.shape[0] == len(lw.weight_names)

        # All values should be within their configured bounds
        for i, name in enumerate(lw.weight_names):
            min_val, max_val, _ = LearnableWeights.WEIGHT_CONFIG[name]
            assert min_val <= tensor[i].item() <= max_val

    def test_learnable_weights_gradient_flow(self):
        """Cover gradient computation through weights."""
        initial = AIWeights()
        lw = LearnableWeights(initial)

        # Compute a simple loss
        weights = lw.get_weights_tensor()
        loss = torch.sum(weights**2)
        loss.backward()

        # Gradients should be computed
        assert lw.raw_weights.grad is not None
        assert not torch.all(lw.raw_weights.grad == 0)

    def test_to_ai_weights_integer_rounding(self):
        """Cover integer rounding in to_ai_weights."""
        initial = AIWeights(
            queen_protection_threshold=3,
            high_heart_threshold=10,
            moon_threat_threshold=10,
        )
        lw = LearnableWeights(initial)

        # Modify raw weights slightly
        with torch.no_grad():
            lw.raw_weights.data += torch.randn_like(lw.raw_weights) * 0.1

        converted = lw.to_ai_weights()

        # Integer fields should still be integers
        assert isinstance(converted.queen_protection_threshold, int)
        assert isinstance(converted.high_heart_threshold, int)
        assert isinstance(converted.moon_threat_threshold, int)


# =============================================================================
# Simulate Game Extended Tests
# =============================================================================


class TestSimulateGameExtended:
    """Extended tests for simulate_game function."""

    def test_simulate_game_4_players_kitty(self):
        """Cover 4-player game (kitty mode not applicable but test default)."""
        weights_4p = [AIWeights()] * 4
        scores = simulate_game(weights_4p, num_players=4)

        assert len(scores) == 4
        # Total points in a 4-player game should be at least 0
        # (could be 26 per round from moon, or normal distribution)
        assert all(s >= 0 for s in scores)

    def test_simulate_game_3_players_kitty(self):
        """Cover 3-player KITTY mode simulation."""
        weights_3p = [AIWeights()] * 3
        scores = simulate_game(
            weights_3p, num_players=3, three_player_mode=ThreePlayerMode.KITTY
        )

        assert len(scores) == 3
        assert all(isinstance(s, (int, float)) for s in scores)

    def test_simulate_game_completes_without_error(self):
        """Ensure simulate_game completes multiple rounds."""
        weights = [AIWeights()] * 4

        # Run multiple simulations to ensure robustness
        for _ in range(3):
            scores = simulate_game(weights, num_players=4)
            assert len(scores) == 4
            # At least one player should reach 100+
            assert any(s >= 100 for s in scores)

    def test_simulate_game_with_custom_weights(self):
        """Cover simulation with customized weights."""
        custom_weights = AIWeights(
            pass_queen_of_spades=150.0,
            discard_queen_of_spades=150.0,
            moon_attempt_threshold=90.0,  # Rarely attempt moon
        )
        weights_list = [custom_weights] * 4

        scores = simulate_game(weights_list, num_players=4)
        assert len(scores) == 4


# =============================================================================
# Compute Reward Extended Tests
# =============================================================================


class TestComputeRewardExtended:
    """Extended tests for compute_reward function."""

    def test_compute_reward_winner(self):
        """Cover reward computation for winner."""
        scores = [50, 60, 70, 100]  # Player 0 wins
        reward = compute_reward(scores, 0)

        # Winner bonus should increase reward
        reward_non_winner = compute_reward(scores, 1)
        assert reward > reward_non_winner

    def test_compute_reward_loser(self):
        """Cover reward computation for loser."""
        scores = [50, 60, 70, 100]  # Player 3 loses
        reward = compute_reward(scores, 3)

        # Loser penalty should decrease reward
        reward_non_loser = compute_reward(scores, 1)
        assert reward < reward_non_loser

    def test_compute_reward_relative_performance(self):
        """Cover relative performance bonus/penalty."""
        scores = [40, 50, 60, 70]

        # Calculate rewards for all players
        rewards = [compute_reward(scores, i) for i in range(4)]

        # Better scores should have better rewards
        assert rewards[0] > rewards[1] > rewards[2] > rewards[3]

    def test_compute_reward_large_score_differences(self):
        """Cover reward with large score differences."""
        scores = [10, 100, 100, 100]

        reward_winner = compute_reward(scores, 0)
        reward_loser = compute_reward(scores, 1)

        # Large difference in rewards expected
        assert reward_winner - reward_loser > 50


# =============================================================================
# HeartsTrainer Extended Tests
# =============================================================================


class TestHeartsTrainerExtended:
    """Extended tests for HeartsTrainer class."""

    def test_trainer_initialization_default_weights(self):
        """Cover trainer initialization with default weights."""
        trainer = HeartsTrainer()

        assert trainer.num_players == 4
        assert trainer.game_mode == GameMode.PLAYER_4
        assert trainer.best_weights is None
        assert trainer.best_avg_score == float("inf")

    def test_trainer_initialization_3_player(self):
        """Cover trainer initialization for 3 players."""
        trainer = HeartsTrainer(
            num_players=3, three_player_mode=ThreePlayerMode.REMOVE_CARD
        )

        assert trainer.num_players == 3
        assert trainer.game_mode == GameMode.PLAYER_3_REMOVE

    def test_trainer_initialization_3_player_kitty(self):
        """Cover trainer initialization for 3-player kitty."""
        trainer = HeartsTrainer(
            num_players=3, three_player_mode=ThreePlayerMode.KITTY
        )

        assert trainer.num_players == 3
        assert trainer.game_mode == GameMode.PLAYER_3_KITTY

    def test_train_epoch_updates_history(self):
        """Cover train_epoch updating history."""
        trainer = HeartsTrainer(initial_weights=AIWeights())

        initial_history_len = len(trainer.history)
        trainer.train_epoch(games_per_epoch=1, verbose=False)

        assert len(trainer.history) == initial_history_len + 1

    def test_train_epoch_tracks_best_weights(self):
        """Cover best weights tracking during training."""
        trainer = HeartsTrainer(initial_weights=AIWeights())

        # Run a few epochs
        for _ in range(3):
            trainer.train_epoch(games_per_epoch=1, verbose=False)

        # Best weights should be set
        assert trainer.best_weights is not None
        assert trainer.best_avg_score < float("inf")

    def test_train_epoch_verbose_output(self):
        """Cover verbose output during training."""
        trainer = HeartsTrainer(initial_weights=AIWeights())

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            trainer.train_epoch(games_per_epoch=1, verbose=True)
            output = mock_stdout.getvalue()

            # Should print epoch info
            assert "Epoch" in output or "avg_score" in output

    def test_print_weights_positive_diff(self):
        """Cover _print_weights with positive difference."""
        trainer = HeartsTrainer()
        weights = AIWeights(pass_queen_of_spades=110.0)  # Higher than default

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            trainer._print_weights(weights)
            output = mock_stdout.getvalue()

            # Should contain positive diff indicator
            assert "+" in output

    def test_print_weights_negative_diff(self):
        """Cover _print_weights with negative difference."""
        trainer = HeartsTrainer()
        weights = AIWeights(pass_queen_of_spades=90.0)  # Lower than default

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            trainer._print_weights(weights)
            output = mock_stdout.getvalue()

            # Should contain negative diff (no + sign for negative)
            assert "-" in output


# =============================================================================
# Training Convergence Tests
# =============================================================================


class TestTrainingConvergence:
    """Tests for training convergence behavior."""

    def test_convergence_with_strict_threshold(self):
        """Cover convergence detection with strict threshold."""
        trainer = HeartsTrainer(initial_weights=AIWeights())

        # Add history with increasing weight changes (not converging)
        trainer.history = [
            TrainingStats(i, 20.0, 0.5, 0.25, 0.5, {}) for i in range(5)
        ]

        # Should not be converged with high weight changes
        assert not trainer.is_converged(window=5, threshold=0.1)

    def test_convergence_early_history(self):
        """Cover convergence check with insufficient history."""
        trainer = HeartsTrainer(initial_weights=AIWeights())

        # Only 2 epochs in history
        trainer.history = [
            TrainingStats(0, 20.0, 0.5, 0.25, 0.01, {}),
            TrainingStats(1, 19.0, 0.6, 0.26, 0.01, {}),
        ]

        # Window of 5 should not be satisfied
        assert not trainer.is_converged(window=5, threshold=0.1)

    def test_train_until_convergence_returns_weights(self):
        """Cover train_until_convergence always returns weights."""
        trainer = HeartsTrainer(initial_weights=AIWeights())

        result = trainer.train_until_convergence(
            max_epochs=2,
            games_per_epoch=1,
            convergence_window=5,
            convergence_threshold=0.1,
            verbose=False,
        )

        assert isinstance(result, AIWeights)


# =============================================================================
# run_learning_mode Tests
# =============================================================================


class TestRunLearningModeExtended:
    """Extended tests for run_learning_mode function."""

    def test_run_learning_mode_output(self):
        """Cover run_learning_mode console output."""
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            result = run_learning_mode(
                max_epochs=1,
                games_per_epoch=1,
                learning_rate=0.01,
                convergence_threshold=0.1,
                num_players=4,
                three_player_mode=ThreePlayerMode.REMOVE_CARD,
            )

            output = mock_stdout.getvalue()
            assert "HEARTS AI LEARNING MODE" in output
            assert "Mode:" in output
            assert isinstance(result, AIWeights)


# =============================================================================
# Training Stats Tests
# =============================================================================


class TestTrainingStats:
    """Tests for TrainingStats dataclass."""

    def test_training_stats_creation(self):
        """Cover TrainingStats instantiation."""
        stats = TrainingStats(
            epoch=5,
            avg_score=25.5,
            avg_reward=-15.0,
            win_rate=0.3,
            weight_change=0.05,
            weights={"pass_queen_of_spades": 100.0},
        )

        assert stats.epoch == 5
        assert stats.avg_score == 25.5
        assert stats.avg_reward == -15.0
        assert stats.win_rate == 0.3
        assert stats.weight_change == 0.05
        assert stats.weights["pass_queen_of_spades"] == 100.0


# =============================================================================
# Edge Case Integration Tests
# =============================================================================


class TestEdgeCaseIntegration:
    """Integration tests for edge cases."""

    def test_simulate_game_handles_all_phases(self):
        """Ensure simulate_game handles all game phases correctly."""
        weights = [AIWeights()] * 4

        # This should go through DEALING -> PASSING -> PLAYING -> ROUND_END multiple times
        scores = simulate_game(weights, num_players=4)

        # Game should complete properly
        assert len(scores) == 4
        assert all(isinstance(s, (int, float)) for s in scores)

    def test_trainer_handles_poor_performance(self):
        """Cover trainer behavior with consistently poor performance."""
        # Use weights that might lead to poor play
        poor_weights = AIWeights(
            pass_queen_of_spades=0.0,  # Never pass QoS
            discard_queen_of_spades=0.0,  # Never discard QoS
        )
        trainer = HeartsTrainer(initial_weights=poor_weights)

        # Should still complete without error
        stats = trainer.train_epoch(games_per_epoch=1, verbose=False)
        assert stats is not None

    def test_learnable_weights_extreme_optimization(self):
        """Cover LearnableWeights under extreme optimization steps."""
        initial = AIWeights()
        lw = LearnableWeights(initial)
        optimizer = torch.optim.SGD(lw.parameters(), lr=1.0)  # High learning rate

        # Take several large optimization steps
        for _ in range(10):
            optimizer.zero_grad()
            weights = lw.get_weights_tensor()
            loss = torch.sum(weights)
            loss.backward()
            optimizer.step()

        # Weights should still be valid (within bounds due to sigmoid)
        converted = lw.to_ai_weights()
        for name in LearnableWeights.WEIGHT_CONFIG:
            min_val, max_val, _ = LearnableWeights.WEIGHT_CONFIG[name]
            value = getattr(converted, name)
            assert min_val <= value <= max_val, f"{name} out of bounds: {value}"
