# Contributing to Hearts AI

Thank you for your interest in contributing to Hearts AI! This document provides guidelines and information for contributors.

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributors of all experience levels.

## How to Contribute

### Reporting Bugs

1. Check existing issues to avoid duplicates
2. Use a clear, descriptive title
3. Describe the steps to reproduce the bug
4. Include expected vs actual behavior
5. Add relevant logs or screenshots

### Suggesting Features

1. Check existing issues for similar suggestions
2. Describe the feature and its use case
3. Explain why it would benefit the project

### Pull Requests

1. Fork the repository
2. Create a feature branch from `master`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes
4. Write or update tests as needed
5. Ensure all tests pass:
   ```bash
   pytest tests/ -v
   ```
6. Ensure code coverage remains high:
   ```bash
   uv run pytest tests/ -v --cov=hearts_ai --cov=hearts_game --cov=hearts_learn
   ```
7. Commit with a clear message:
   ```bash
   git commit -m "Add feature: description of changes"
   ```
8. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
9. Open a Pull Request

## Development Setup

### Prerequisites

- Python 3.14+
- uv (https://github.com/indygreg/uv) or pip

### Installation

```bash
# Clone your fork
git clone https://github.com/yourusername/hearts.git
cd hearts

# Install with uv (recommended)
uv sync

# Alternatively, using pip:
# pip install -r requirements.txt
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=hearts_ai --cov=hearts_game --cov=hearts_learn --cov-report=term-missing

# Run specific test file
pytest tests/test_ai.py -v

# Run specific test
pytest tests/test_ai.py::test_ai_weight_selection -v
```

## Code Style

### General Guidelines

- Use clear, descriptive variable and function names
- Add docstrings to public functions and classes
- Keep functions focused and reasonably sized
- Use type hints where appropriate

### Python Style

- Follow PEP 8 guidelines
- Use 4 spaces for indentation
- Maximum line length: 88 characters (Black formatter default)
- Use snake_case for functions and variables
- Use PascalCase for classes

### Example

```python
def calculate_pass_priority(
    card: Card,
    hand: list[Card],
    weights: AIWeights,
) -> float:
    """
    Calculate the priority for passing a specific card.

    Args:
        card: The card to evaluate
        hand: The player's current hand
        weights: AI weight configuration

    Returns:
        Priority score (higher = more likely to pass)
    """
    priority = weights.pass_base_priority

    if card.is_queen_of_spades:
        priority += weights.pass_queen_of_spades

    return priority
```

## Project Structure

```
hearts/
├── hearts_game.py      # Game rules and state - modify for rule changes
├── hearts_ai.py        # AI logic - modify for strategy improvements
├── hearts_learn.py     # Training system - modify for learning improvements
├── hearts_gui.py       # GUI - modify for UI changes
├── weights.json        # AI weights - modify for strategy improvements
├── tests/              # Test suite - add tests for new features
│   ├── conftest.py     # Shared fixtures
│   ├── test_game.py    # Game logic tests
│   ├── test_ai.py      # AI behavior tests
│   └── ...
└── .github/
    └── assistant.json  # AI context - update when adding major features
```

## Areas for Contribution

### Good First Issues

- Add more test cases for edge conditions
- Improve documentation and docstrings
- Fix typos or clarify comments

### Feature Ideas

- **AI Improvements**:
  - New heuristic weights for specific strategies
  - Better moon shooting detection
  - Opponent modeling

- **Training Enhancements**:
  - Alternative optimization algorithms
  - Tournament-style training
  - Curriculum learning

- **GUI Improvements**:
  - Card animations
  - Sound effects
  - Statistics display

- **Game Variants**:
  - Omnibus Hearts (10 of diamonds = -10)
  - Black Maria variant
  - Custom scoring rules

## Testing Guidelines

### Test Coverage

- Maintain nearly 100% coverage on core modules (`hearts_ai.py`, `hearts_game.py`)
- Add tests for any new functionality
- Test both happy paths and edge cases

### Test Structure

```python
class TestFeatureName:
    """Tests for specific feature."""

    def test_normal_case(self):
        """Test the expected behavior."""
        # Arrange
        ai = HeartsAI()

        # Act
        result = ai.some_method()

        # Assert
        assert result == expected

    def test_edge_case(self):
        """Test edge case behavior."""
        ...
```

### Running Specific Tests

```bash
# Run tests matching a pattern
pytest tests/ -v -k "moon"

# Run tests in a specific class
pytest tests/test_ai.py::TestMoonThreatLead -v

# Run with verbose output
pytest tests/ -vv
```

## Commit Messages

Use clear, descriptive commit messages:

```
type: short description

Longer description if needed, explaining:
- What was changed
- Why it was changed
- Any breaking changes
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `style`: Formatting changes
- `perf`: Performance improvements

## Questions?

Feel free to open an issue for any questions about contributing.
