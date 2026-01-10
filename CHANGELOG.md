# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-01-10

### Added
- **Advanced AI Strategies**:
    - Finesse (high ducking) logic to draw out high cards.
    - Bleeding Spades strategy when holding a protected Queen of Spades.
    - Advanced Moon Offense (dropping low hearts early with suit control).
    - Strategic 2 of Clubs and Ace of Clubs control during passing.
    - Decoy passes to obscure hand strength.
    - Void signaling avoidance by penalizing passing the last card of a suit.
- **Opponent Tracking**: Expanded `CardTracker` to monitor passed cards and estimate high card holding probabilities.
- **Improved Weights**: New set of optimized weights in `improved_weights.py` trained using reinforcement learning on the new strategies.
- **Evaluation Tools**: `benchmark_ai.py` for comparing different weight sets through automated tournaments.
- **GitHub Documentation**: Comprehensive `README.md` updates, `CONTRIBUTING.md`, `LICENSE`, and `CHANGELOG.md`.

### Fixed
- Edge case in passing logic where identical cards could be selected.
- Scoping error in AI discard logic during moon shooting scenarios.

## [0.1.0] - 2025-12-25

### Added
- Initial release of Hearts AI.
- Core game engine and rules.
- Heuristic-based AI with 60+ configurable weights.
- PyTorch training system.
- Pygame-based GUI.
- Standard unit test suite.
