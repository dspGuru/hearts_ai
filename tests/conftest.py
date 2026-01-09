import pytest
from hearts_game import HeartsGame, ThreePlayerMode, Card, Suit, Rank


@pytest.fixture
def game_4p():
    """A 4-player game in the dealing phase."""
    return HeartsGame(["P0", "P1", "P2", "P3"])


@pytest.fixture
def game_3p_remove():
    """A 3-player game (REMOVE_CARD) in the dealing phase."""
    return HeartsGame(["P0", "P1", "P2"], three_player_mode=ThreePlayerMode.REMOVE_CARD)


@pytest.fixture
def game_3p_kitty():
    """A 3-player game (KITTY) in the dealing phase."""
    return HeartsGame(["P0", "P1", "P2"], three_player_mode=ThreePlayerMode.KITTY)


@pytest.fixture
def sample_cards():
    """A few sample cards for testing."""
    return [
        Card(Suit.CLUBS, Rank.TWO),
        Card(Suit.SPADES, Rank.QUEEN),
        Card(Suit.HEARTS, Rank.ACE),
        Card(Suit.DIAMONDS, Rank.TEN),
    ]
