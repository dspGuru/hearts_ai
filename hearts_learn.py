"""
Learning module for Hearts AI.

Uses PyTorch to optimize AI heuristic weights through self-play.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass, fields
from typing import Optional
import random
import time

from hearts_game import (
    Card,
    Suit,
    Rank,
    HeartsGame,
    Trick,
    ThreePlayerMode,
    GameMode,
    GamePhase,
)
from hearts_ai import AIWeights, HeartsAI, DEFAULT_WEIGHTS


class LearnableWeights(nn.Module):
    """
    PyTorch module that wraps AIWeights as learnable parameters.
    """

    # Define which weights are learnable and their bounds
    WEIGHT_CONFIG = {
        # (min, max, is_integer)
        "pass_queen_of_spades": (50.0, 150.0, False),
        "queen_protection_threshold": (1, 5, True),
        "pass_to_void_suit": (40.0, 120.0, False),
        "void_minor_suit_bonus": (0.0, 20.0, False),
        "pass_high_hearts": (30.0, 100.0, False),
        "high_heart_threshold": (8, 12, True),
        "pass_high_cards": (20.0, 80.0, False),
        "high_card_threshold": (10, 14, True),
        "pass_base_priority": (-10.0, 10.0, False),
        "lead_clubs_priority": (20.0, 60.0, False),
        "lead_diamonds_priority": (15.0, 50.0, False),
        "lead_spades_priority": (5.0, 40.0, False),
        "lead_hearts_priority": (0.0, 30.0, False),
        "lead_low_card_preference": (0.5, 5.0, False),
        "discard_queen_of_spades": (60.0, 150.0, False),
        "discard_hearts": (40.0, 120.0, False),
        "discard_dangerous_spades": (30.0, 100.0, False),
        "discard_high_cards": (20.0, 80.0, False),
        "discard_rank_multiplier": (0.5, 3.0, False),
        "high_duck_preference": (0.5, 2.0, False),
        "low_win_preference": (0.5, 2.0, False),
        "take_safe_trick_preference": (0.5, 2.0, False),
        "moon_threat_threshold": (5, 15, True),
        "moon_block_priority": (20.0, 80.0, False),
    }

    def __init__(self, initial_weights: AIWeights):
        super().__init__()

        if initial_weights is None:
            raise ValueError("initial_weights must be provided")

        # Create learnable parameters for each weight
        self.weight_names = list(self.WEIGHT_CONFIG.keys())

        # Store as a single parameter tensor for easier optimization
        initial_values = []
        for name in self.weight_names:
            value = getattr(initial_weights, name)
            min_val, max_val, _ = self.WEIGHT_CONFIG[name]
            # Normalize to [0, 1] range for better optimization
            normalized = (value - min_val) / (max_val - min_val)
            initial_values.append(normalized)

        initial_values_tensor = torch.tensor(initial_values, dtype=torch.float32)
        # Apply inverse sigmoid (logit) so that sigmoid(raw_weights) starts at initial_values
        # Clamp to avoid infinity at 0 or 1
        eps = 1e-6
        initial_values_tensor = torch.clamp(initial_values_tensor, eps, 1.0 - eps)
        logit_values = torch.log(initial_values_tensor / (1.0 - initial_values_tensor))

        self.raw_weights = nn.Parameter(logit_values)

    def get_weights_tensor(self) -> torch.Tensor:
        """Get the actual weight values (denormalized and clamped)."""
        weights = torch.sigmoid(self.raw_weights)  # Ensure [0, 1]

        actual_weights = []
        for i, name in enumerate(self.weight_names):
            min_val, max_val, is_int = self.WEIGHT_CONFIG[name]
            value = weights[i] * (max_val - min_val) + min_val
            actual_weights.append(value)

        return torch.stack(actual_weights)

    def to_ai_weights(self) -> AIWeights:
        """Convert current parameters to AIWeights object."""
        weights_tensor = self.get_weights_tensor()

        kwargs = {}
        for i, name in enumerate(self.weight_names):
            _, _, is_int = self.WEIGHT_CONFIG[name]
            value = weights_tensor[i].item()
            if is_int:
                value = int(round(value))
            kwargs[name] = value

        return AIWeights(**kwargs)

    def get_weight_dict(self) -> dict[str, float]:
        """Get weights as a dictionary."""
        weights_tensor = self.get_weights_tensor()
        result = {}
        for i, name in enumerate(self.weight_names):
            _, _, is_int = self.WEIGHT_CONFIG[name]
            value = weights_tensor[i].item()
            if is_int:
                value = int(round(value))
            result[name] = value
        return result


def simulate_game(
    weights_list: list[AIWeights],
    num_players: int = 4,
    three_player_mode: ThreePlayerMode = ThreePlayerMode.REMOVE_CARD,
) -> list[int]:
    """
    Simulate a complete Hearts game with given AI weights.

    Args:
        weights_list: List of AIWeights for each player
        num_players: Number of players (3 or 4)
        three_player_mode: How to handle extra card in 3p games

    Returns:
        List of final scores for each player
    """
    names = [f"Player{i}" for i in range(num_players)]
    game = HeartsGame(names, three_player_mode)

    # Create AI instances with specific weights
    ais = [HeartsAI(weights=w) for w in weights_list]
    for ai in ais:
        ai.reset_round(num_players)

    while not game.is_game_over():
        game.start_round()

        # Reset AI tracking for new round
        for ai in ais:
            ai.reset_round(num_players)

        # Passing phase
        if game.phase == GamePhase.PASSING:
            for i, player in enumerate(game.players):
                cards_to_pass = ais[i].select_pass_cards(
                    player.hand, game.pass_direction.name
                )
                game.set_pass_cards(i, cards_to_pass)
            game.execute_pass()

        # Playing phase
        while game.phase == GamePhase.PLAYING:
            current_idx = game.current_player_index
            player = game.players[current_idx]
            valid_plays = game.get_valid_plays(current_idx)

            if not valid_plays:
                break

            card = ais[current_idx].select_play(
                hand=player.hand,
                valid_plays=valid_plays,
                trick=game.current_trick,
                player_index=current_idx,
                num_players=num_players,
                hearts_broken=game.hearts_broken,
            )

            result = game.play_card(current_idx, card)

            if result["trick_complete"]:
                # Record trick for all AIs
                for ai in ais:
                    ai.record_trick(
                        result["trick_cards"],
                        result["trick_winner"],
                        result["trick_info"],
                    )

    return [p.total_score for p in game.players]


def compute_reward(scores: list[int], player_idx: int) -> float:
    """
    Compute reward for a player based on game outcome.

    Lower score is better in Hearts, so we use negative score.
    Also consider relative ranking.
    """
    my_score = scores[player_idx]

    # Base reward: negative of score (lower is better)
    reward = -my_score

    # Bonus for winning (lowest score)
    if my_score == min(scores):
        reward += 20

    # Penalty for losing (highest score)
    if my_score == max(scores):
        reward -= 20

    # Relative performance bonus
    avg_score = sum(scores) / len(scores)
    reward += (avg_score - my_score) * 0.5

    return reward


@dataclass
class TrainingStats:
    """Statistics from training."""

    epoch: int
    avg_score: float
    avg_reward: float
    win_rate: float
    weight_change: float
    weights: dict[str, float]


class HeartsTrainer:
    """
    Trainer for optimizing Hearts AI weights through self-play.
    """

    def __init__(
        self,
        initial_weights: AIWeights = None,
        learning_rate: float = 0.01,
        num_players: int = 4,
        three_player_mode: ThreePlayerMode = ThreePlayerMode.REMOVE_CARD,
    ):
        self.num_players = num_players
        self.three_player_mode = three_player_mode
        self.game_mode = GameMode.from_settings(num_players, three_player_mode)

        if initial_weights is None:
            initial_weights = DEFAULT_WEIGHTS.get(self.game_mode, AIWeights())

        self.learnable = LearnableWeights(initial_weights)
        self.optimizer = optim.Adam(self.learnable.parameters(), lr=learning_rate)

        self.history: list[TrainingStats] = []
        self.best_weights: Optional[AIWeights] = None
        self.best_avg_score = float("inf")

    def train_epoch(
        self, games_per_epoch: int = 20, verbose: bool = False
    ) -> TrainingStats:
        """
        Run one training epoch.

        The learnable AI plays against copies of itself (or baseline).
        We use REINFORCE-style policy gradient to update weights.
        """
        epoch = len(self.history)

        # Get current weights
        current_weights = self.learnable.to_ai_weights()

        # Track scores and rewards
        all_scores = []
        all_rewards = []
        wins = 0

        # Run games
        for game_num in range(games_per_epoch):
            # All players use the same weights (self-play)
            weights_list = [current_weights] * self.num_players

            scores = simulate_game(
                weights_list, self.num_players, self.three_player_mode
            )

            # Evaluate from perspective of player 0
            reward = compute_reward(scores, 0)

            all_scores.append(scores[0])
            all_rewards.append(reward)

            if scores[0] == min(scores):
                wins += 1

        # Compute statistics
        avg_score = np.mean(all_scores)
        avg_reward = np.mean(all_rewards)
        win_rate = wins / games_per_epoch

        # Compute loss and update weights
        # We use the negative reward as loss (want to maximize reward)
        self.optimizer.zero_grad()

        # Simple policy gradient: adjust weights in direction that improves reward
        # We use the reward as a scaling factor for the gradient
        reward_tensor = torch.tensor(avg_reward, dtype=torch.float32)

        # Get weights and compute a simple loss
        weights_tensor = self.learnable.get_weights_tensor()

        # Loss is negative reward (we want to maximize reward)
        # We add a small regularization term to prevent extreme values
        loss = -reward_tensor + 0.01 * torch.sum(weights_tensor**2) * 0.001

        # Compute gradient through the reward signal
        # Since reward depends on game outcomes (non-differentiable),
        # we use the reward to scale random perturbations
        noise = torch.randn_like(self.learnable.raw_weights) * 0.1
        perturbed_loss = loss + torch.sum(noise * self.learnable.raw_weights) * (
            avg_reward / 100
        )

        perturbed_loss.backward()

        # Clip gradients to prevent instability
        torch.nn.utils.clip_grad_norm_(self.learnable.parameters(), 1.0)

        # Save old weights for convergence check
        old_weights = self.learnable.get_weight_dict()

        self.optimizer.step()

        # Compute weight change
        new_weights = self.learnable.get_weight_dict()
        weight_change = sum(
            abs(new_weights[k] - old_weights[k]) for k in old_weights
        ) / len(old_weights)

        # Track best weights
        if avg_score < self.best_avg_score:
            self.best_avg_score = avg_score
            self.best_weights = self.learnable.to_ai_weights()

        stats = TrainingStats(
            epoch=epoch,
            avg_score=avg_score,
            avg_reward=avg_reward,
            win_rate=win_rate,
            weight_change=weight_change,
            weights=new_weights,
        )
        self.history.append(stats)

        if verbose:
            print(
                f"Epoch {epoch:3d}: avg_score={avg_score:.1f}, "
                f"win_rate={win_rate:.1%}, weight_change={weight_change:.4f}"
            )

        return stats

    def is_converged(self, window: int = 10, threshold: float = 0.1) -> bool:
        """
        Check if training has converged.

        Convergence is detected when the average weight change
        over the last `window` epochs falls below `threshold`.
        """
        if len(self.history) < window:
            return False

        recent_changes = [s.weight_change for s in self.history[-window:]]
        avg_change = np.mean(recent_changes)

        return avg_change < threshold

    def train_until_convergence(
        self,
        max_epochs: int = 100,
        games_per_epoch: int = 20,
        convergence_window: int = 10,
        convergence_threshold: float = 0.1,
        verbose: bool = True,
    ) -> AIWeights:
        """
        Train until weights converge or max epochs reached.

        Returns the best weights found.
        """
        if verbose:
            print("Starting training...")
            print(f"  Max epochs: {max_epochs}")
            print(f"  Games per epoch: {games_per_epoch}")
            print(
                f"  Convergence: window={convergence_window}, threshold={convergence_threshold}"
            )
            print()

        start_time = time.time()

        for epoch in range(max_epochs):
            stats = self.train_epoch(games_per_epoch, verbose)

            if self.is_converged(convergence_window, convergence_threshold):
                if verbose:
                    print(f"\nConverged after {epoch + 1} epochs!")
                break
        else:
            if verbose:
                print(f"\nReached max epochs ({max_epochs})")

        elapsed = time.time() - start_time

        if verbose:
            print(f"\nTraining completed in {elapsed:.1f} seconds")
            print(f"Best average score: {self.best_avg_score:.1f}")
            print("\nFinal weights:")
            self._print_weights(self.best_weights)

        return self.best_weights

    def _print_weights(self, weights: AIWeights):
        """Print weights in a readable format."""
        # Get baseline weights for this mode
        baseline = DEFAULT_WEIGHTS.get(self.game_mode, AIWeights())

        for name in self.learnable.weight_names:
            value = getattr(weights, name)
            default = getattr(baseline, name)
            diff = value - default
            diff_str = f"+{diff:.1f}" if diff > 0 else f"{diff:.1f}"
            print(f"  {name}: {value:.1f} ({diff_str} from default)")


def run_learning_mode(
    max_epochs: int = 100,
    games_per_epoch: int = 20,
    learning_rate: float = 0.01,
    convergence_threshold: float = 0.1,
    num_players: int = 4,
    three_player_mode: ThreePlayerMode = ThreePlayerMode.REMOVE_CARD,
) -> AIWeights:
    """
    Run the learning mode and return optimized weights.
    """
    print("=" * 60)
    print("HEARTS AI LEARNING MODE")
    print(f"Mode: {num_players} players, {three_player_mode.name}")
    print("=" * 60)
    print()

    trainer = HeartsTrainer(
        learning_rate=learning_rate,
        num_players=num_players,
        three_player_mode=three_player_mode,
    )

    best_weights = trainer.train_until_convergence(
        max_epochs=max_epochs,
        games_per_epoch=games_per_epoch,
        convergence_threshold=convergence_threshold,
        verbose=True,
    )

    return best_weights


def main():
    """Entry point for learning mode."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Hearts AI weights")
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=100,
        help="Maximum training epochs (default: 100)",
    )
    parser.add_argument(
        "--games", "-g", type=int, default=20, help="Games per epoch (default: 20)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.1,
        help="Convergence threshold (default: 0.1)",
    )
    parser.add_argument(
        "--players",
        "-p",
        type=int,
        choices=[3, 4],
        default=4,
        help="Number of players (default: 4)",
    )
    parser.add_argument(
        "--kitty", action="store_true", help="Use kitty mode for 3-player game"
    )
    parser.add_argument(
        "--all-modes",
        action="store_true",
        help="Train for all three game modes sequentially",
    )
    parser.add_argument("--save", "-s", type=str, help="Save learned weights to file")

    args = parser.parse_args()

    if args.all_modes:
        training_configs = [
            (4, ThreePlayerMode.REMOVE_CARD),
            (3, ThreePlayerMode.REMOVE_CARD),
            (3, ThreePlayerMode.KITTY),
        ]
    else:
        three_player_mode = ThreePlayerMode.REMOVE_CARD
        if args.players == 3 and args.kitty:
            three_player_mode = ThreePlayerMode.KITTY
        training_configs = [(args.players, three_player_mode)]

    all_best_weights: dict[GameMode, AIWeights] = {}

    for num_players, tp_mode in training_configs:
        best_weights = run_learning_mode(
            max_epochs=args.epochs,
            games_per_epoch=args.games,
            learning_rate=args.lr,
            convergence_threshold=args.threshold,
            num_players=num_players,
            three_player_mode=tp_mode,
        )
        mode = GameMode.from_settings(num_players, tp_mode)
        all_best_weights[mode] = best_weights

    if args.save:
        # Save weights to Python file using the new multi-mode dictionary format
        with open(args.save, "w") as f:
            f.write("# Learned Hearts AI weights\n")
            f.write("from hearts_ai import AIWeights\n")
            f.write("from hearts_game import GameMode\n\n")

            # Helper to write one AIWeights instance
            def write_weights(w: AIWeights, indent: str):
                f.write(f"{indent}AIWeights(\n")
                for name in LearnableWeights.WEIGHT_CONFIG:
                    value = getattr(w, name)
                    _, _, is_int = LearnableWeights.WEIGHT_CONFIG[name]
                    if is_int:
                        f.write(f"{indent}    {name}={int(value)},\n")
                    else:
                        f.write(f"{indent}    {name}={value:.2f},\n")
                f.write(f"{indent}),\n")

            f.write("LEARNED_WEIGHTS = {\n")
            # We want to ensure all modes are in the dict, even if not trained
            all_possible_modes = [
                GameMode.PLAYER_4,
                GameMode.PLAYER_3_REMOVE,
                GameMode.PLAYER_3_KITTY,
            ]
            for m in all_possible_modes:
                f.write(f"    GameMode.{m.name}: ")
                if m in all_best_weights:
                    write_weights(all_best_weights[m], "    ")
                else:
                    # Use default weights if not trained this time
                    f.write("AIWeights(),\n")
            f.write("}\n")
        print(f"\nWeights saved to {args.save}")


if __name__ == "__main__":
    main()
