# Hearts AI

A complete implementation of the Hearts card game with an AI player that uses heuristic-based decision-making optimized through reinforcement learning.

This project was created entirely via "vibe coding" using AI assistants. It was created as a proof of concept for the viability of AI assistants in development of a straight-forward but non-trivial game which includes "AI" players. The
initial version does not include a single line of code written by a human.

The original version was created over the course of two days with about eight hours
of human involvement which included guiding the AI assistants in creating the initial game concept and options, implementing the core game engine, refining the GUI, and training the AI players, and writing some of the documentation.

This project also demonstrates the use of PyTorch, Pygame, and pytest for machine learning, game development, and testing.

## Features

- **Full Hearts Implementation**: Complete game rules including passing, playing, and scoring
- **Multiple Game Modes**: 4-player standard, 3-player (removal or kitty)
- **Advanced AI Heuristics**: 80+ configurable weights covering nuanced strategies (Finesse, Bleeding Spades, Moon Offense, etc.)
- **Reinforcement Learning**: PyTorch-based weight optimization through continuous self-play
- **Benchmark Suite**: Compare AI versions and validate new strategies
- **Interactive GUI**: Pygame-based interface with auto-play and delay controls
- **High Test Coverage**: Robust unit test suite (90%+ coverage)

## Installation

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Setup with uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/hearts.git
cd hearts

# Install dependencies and create environment
uv sync
```

### Setup with pip

```bash
git clone https://github.com/yourusername/hearts.git
cd hearts
pip install -r requirements.txt
```

## Quick Start

### Play Against AI

```bash
python hearts_gui.py
```

### Watch AI vs AI

```bash
python hearts_gui.py --auto
```

### Train AI Weights

```bash
# Train for all game modes
python hearts_learn.py --all-modes --epochs 100 --games 50

# Train for specific mode
python hearts_learn.py --players 4 --epochs 50 --games 30

# Save trained weights
python hearts_learn.py --all-modes --save my_weights.py
```

### Verify AI Performance

```bash
python verify_heuristics.py
```

## Project Structure

```
hearts/
├── hearts_game.py       # Core game engine and rules
├── hearts_ai.py         # AI player with configurable heuristics
├── hearts_learn.py      # PyTorch-based training system
├── hearts_gui.py        # Pygame graphical interface
├── learned_weights.py   # Pre-trained weight configurations
├── improved_weights.py  # Latest optimized weights
├── verify_heuristics.py # AI performance verification
├── tests/               # Test suite (100% coverage)
│   ├── conftest.py
│   ├── test_game.py
│   ├── test_ai.py
│   ├── test_learn.py
│   ├── test_advanced_strategies.py
│   └── test_coverage_gaps.py
└── .github/
    └── assistant.json   # AI assistant context file
```

## Game Rules

### Objective
Avoid taking points. The player with the lowest score when any player reaches 100 points wins.

### Scoring
| Card | Points |
|------|--------|
| Each Heart | 1 |
| Queen of Spades | 13 |
| **Shooting the Moon** | If one player takes all 26 points, they score 0 and others score 26 |

### Passing
Cards are passed each round in rotation:
- **4 players**: Left, Right, Across, None (repeat)
- **3 players**: Left, Right, None (repeat)

### Play Rules
- 2 of Clubs leads the first trick
- Must follow suit if possible
- Hearts cannot be led until "broken" (played when unable to follow suit)
- No points may be played on the first trick (unless unavoidable)

## AI Strategy

The AI uses a weighted heuristic system covering:

### Passing Strategy
- Queen of Spades danger assessment
- Suit voiding for discard opportunities
- High card evaluation
- Strategic 2/Ace of Clubs control

### Playing Strategy
- Position-aware play (2nd hand low, 3rd hand high, last position optimization)
- Moon threat detection and blocking
- Queen of Spades flushing
- Exit card preservation
- Finesse opportunities

### Key Weight Categories

| Category | Description |
|----------|-------------|
| `pass_*` | Card passing decisions |
| `lead_*` | Leading trick priorities |
| `discard_*` | Discarding when void |
| `moon_*` | Shooting/blocking the moon |

See `hearts_ai.py` for the full `AIWeights` dataclass with 60+ parameters.

## CLI Reference

### GUI (`hearts_gui.py`)

```bash
python hearts_gui.py [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--players, -p` | 4 | Number of players (3 or 4) |
| `--kitty` | false | Enable kitty mode for 3-player |
| `--auto` | false | Watch mode (all AI players) |
| `--delay` | 1.0 | Seconds between AI moves in auto mode |

### Training (`hearts_learn.py`)

```bash
python hearts_learn.py [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs, -e` | 100 | Maximum training epochs |
| `--games, -g` | 20 | Games per epoch |
| `--lr` | 0.01 | Learning rate |
| `--threshold, -t` | 0.1 | Convergence threshold |
| `--players, -p` | 4 | Number of players (3 or 4) |
| `--kitty` | false | Use kitty mode for 3-player |
| `--all-modes` | false | Train all game modes |
| `--save, -s` | - | Save weights to file |

### Verification (`verify_heuristics.py`)

```bash
python verify_heuristics.py [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--players, -p` | all | Number of players to test |
| `--kitty` | false | Use kitty mode for 3-player |
| `--games, -g` | 50 | Games to simulate |

## API Usage

### Basic Game

```python
from hearts_game import HeartsGame, Card, Suit, Rank

# Create a 4-player game
game = HeartsGame(["Alice", "Bob", "Carol", "Dave"])
game.start_round()

# Get valid plays for current player
valid = game.get_valid_plays(game.current_player_index)

# Play a card
result = game.play_card(player_index, card)
```

### Using the AI

```python
from hearts_ai import HeartsAI, AIWeights

# Create AI with default weights
ai = HeartsAI()
ai.reset_round(num_players=4)

# Select cards to pass
cards_to_pass = ai.select_pass_cards(hand, "LEFT")

# Select a card to play
card = ai.select_play(
    hand=player.hand,
    valid_plays=valid_plays,
    trick=current_trick,
    player_index=0,
    num_players=4,
    hearts_broken=game.hearts_broken
)

# Record completed trick for tracking
ai.record_trick(trick_cards, winner_index)
```

### Custom Weights

```python
from hearts_ai import AIWeights, HeartsAI

# Create custom weights
weights = AIWeights(
    pass_queen_of_spades=120.0,  # More aggressive QoS passing
    moon_block_priority=80.0,    # Prioritize blocking moon attempts
)

ai = HeartsAI(weights=weights)
```

## Training System

The training system uses REINFORCE-style policy gradients with self-play:

```python
from hearts_learn import HeartsTrainer

trainer = HeartsTrainer(learning_rate=0.01)
best_weights = trainer.train_until_convergence(
    max_epochs=100,
    games_per_epoch=50
)
```

### Training Results

| Mode | Best Avg Score | Typical Convergence |
|------|---------------|---------------------|
| 4-player | ~68 | 100-150 epochs |
| 3-player (remove) | ~77 | 20-50 epochs |
| 3-player (kitty) | ~72 | 60-100 epochs |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=hearts_ai --cov=hearts_game --cov=hearts_learn

# Run specific test file
pytest tests/test_ai.py -v
```

Current coverage: **100%** on `hearts_ai.py` and `hearts_game.py`

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/ -v`)
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Initial implementation by Claude Code (Anthropic)
- Improvements and tests by Gemini 3 Flash
- Optimized weights through reinforcement learning self-play
