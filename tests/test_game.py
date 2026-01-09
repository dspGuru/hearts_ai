import pytest
import random
from unittest.mock import patch
from hearts_game import (
    HeartsGame,
    Card,
    Suit,
    Rank,
    GamePhase,
    ThreePlayerMode,
    Player,
    Trick,
    PassDirection,
    GameMode,
)


def test_string_representations():
    """Verify __str__ and __repr__ for various classes."""
    suit = Suit.HEARTS
    rank = Rank.ACE
    card = Card(suit, rank)

    assert str(suit) == "♥"
    assert str(rank) == "A"
    assert str(card) == "A♥"
    assert repr(card) == "A♥"


def test_card_points():
    """Verify card point values."""
    qos = Card(Suit.SPADES, Rank.QUEEN)
    assert qos.points == 13
    assert qos.is_queen_of_spades

    two_clubs = Card(Suit.CLUBS, Rank.TWO)
    assert two_clubs.is_two_of_clubs

    heart = Card(Suit.HEARTS, Rank.TWO)
    assert heart.points == 1

    club = Card(Suit.CLUBS, Rank.ACE)
    assert club.points == 0


def test_game_initialization_4p(game_4p):
    """Test 4-player game initialization."""
    assert len(game_4p.players) == 4
    assert game_4p.phase == GamePhase.DEALING
    game_4p.start_round()
    assert game_4p.phase == GamePhase.PASSING
    for p in game_4p.players:
        assert len(p.hand) == 13


def test_game_initialization_3p_remove(game_3p_remove):
    """Test 3-player remove mode initialization."""
    assert len(game_3p_remove.players) == 3
    game_3p_remove.start_round()
    assert game_3p_remove.phase == GamePhase.PASSING
    for p in game_3p_remove.players:
        assert len(p.hand) == 17
    # 2 of Diamonds should be missing
    two_diamonds = Card(Suit.DIAMONDS, Rank.TWO)
    for p in game_3p_remove.players:
        assert not p.has_card(two_diamonds)


def test_game_initialization_3p_kitty(game_3p_kitty):
    """Test 3-player kitty mode initialization."""
    assert len(game_3p_kitty.players) == 3
    game_3p_kitty.start_round()
    assert game_3p_kitty.phase == GamePhase.PASSING
    for p in game_3p_kitty.players:
        assert len(p.hand) == 17

    assert game_3p_kitty.kitty_card is not None


def test_valid_play_mechanics(game_4p):
    """Test basic trick play mechanics."""
    game_4p.start_round()
    # Skip passing for simplicity in this test by manually setting phase
    game_4p.phase = GamePhase.PLAYING
    game_4p._start_first_trick()

    start_player_idx = game_4p.current_player_index
    two_clubs = Card(Suit.CLUBS, Rank.TWO)
    assert game_4p.players[start_player_idx].has_card(two_clubs)

    valid = game_4p.get_valid_plays(start_player_idx)
    assert two_clubs in valid
    # Start of round: MUST play 2 of Clubs
    assert len(valid) == 1

    game_4p.play_card(start_player_idx, two_clubs)
    assert len(game_4p.current_trick.cards) == 1
    assert game_4p.current_player_index == (start_player_idx + 1) % 4


def test_shooting_the_moon():
    """Verify shooting the moon scoring logic."""
    game = HeartsGame(["P0", "P1", "P2", "P3"])
    # Give player 0 all hearts and QoS
    p0 = game.players[0]
    for r in range(2, 15):
        p0.take_trick([Card(Suit.HEARTS, Rank(r))])
    p0.take_trick([Card(Suit.SPADES, Rank.QUEEN)])

    # Manually trigger end round
    game._end_round()

    assert game.players[0].round_score == 0
    assert game.players[1].round_score == 26
    assert game.players[2].round_score == 26
    assert game.players[3].round_score == 26


def test_player_heuristics():
    """Verify player hand querying methods."""
    p = Player("Alice")
    cards = [
        Card(Suit.CLUBS, Rank.TWO),
        Card(Suit.HEARTS, Rank.THREE),
        Card(Suit.HEARTS, Rank.FOUR),
    ]
    p.receive_cards(cards)

    assert p.has_suit(Suit.CLUBS)
    assert p.has_suit(Suit.HEARTS)
    assert not p.has_suit(Suit.DIAMONDS)

    assert not p.has_only_hearts()
    p.play_card(cards[0])
    assert p.has_only_hearts()

    hearts = p.get_cards_of_suit(Suit.HEARTS)
    assert len(hearts) == 2
    assert all(c.suit == Suit.HEARTS for c in hearts)


def test_game_errors_and_state():
    """Verify incorrect usage and state reporting."""
    with pytest.raises(ValueError):
        HeartsGame(["Alice", "Bob"])  # Too few players

    game = HeartsGame(["P0", "P1", "P2", "P3"])
    state = game.get_game_state()
    assert state["phase"] == "DEALING"
    assert state["num_players"] == 4

    scores = game.get_scores()
    assert len(scores) == 4
    assert all(s == 0 for s in scores.values())

    round_scores = game.get_round_scores()
    assert len(round_scores) == 4
    assert all(s == 0 for s in round_scores.values())


def test_invalid_plays(game_4p):
    """Verify enforcement of game rules."""
    game_4p.start_round()
    game_4p.phase = GamePhase.PLAYING
    game_4p._start_first_trick()

    p_idx = game_4p.current_player_index
    # Try to play a card not in hand
    not_in_hand = Card(Suit.DIAMONDS, Rank.ACE)
    # Ensure it's actually not in hand
    if game_4p.players[p_idx].has_card(not_in_hand):
        not_in_hand = Card(Suit.DIAMONDS, Rank.TWO)

    res = game_4p.play_card(p_idx, not_in_hand)
    assert not res["valid"]
    assert res["error"] is not None
    assert "not a valid play" in res["error"].lower()

    # Test invalid player name list
    with pytest.raises(ValueError):
        HeartsGame(["P1", "P2", "P3", "P4", "P5"])


def test_game_mode_coverage():
    """Verify GameMode.from_settings coverage."""
    assert GameMode.from_settings(4) == GameMode.PLAYER_4
    assert (
        GameMode.from_settings(3, ThreePlayerMode.REMOVE_CARD)
        == GameMode.PLAYER_3_REMOVE
    )
    assert GameMode.from_settings(3, ThreePlayerMode.KITTY) == GameMode.PLAYER_3_KITTY
    assert GameMode.from_settings(5) == GameMode.PLAYER_4  # Default


def test_missing_two_of_clubs():
    """Verify error when 2 of clubs is missing from all hands."""
    game = HeartsGame(["P0", "P1", "P2", "P3"])
    # Empty hands
    game.phase = GamePhase.PLAYING
    with pytest.raises(RuntimeError):
        game._start_first_trick()


def test_player_error_cases():
    """Verify Player error handling."""
    p = Player("Alice")
    p.receive_cards([Card(Suit.CLUBS, Rank.TWO)])
    with pytest.raises(ValueError):
        p.play_card(Card(Suit.SPADES, Rank.ACE))  # Not in hand


def test_trick_error_cases():
    """Verify Trick error handling."""
    t = Trick(lead_player_index=0)
    with pytest.raises(ValueError):
        t.get_winner()  # Empty trick
    assert t.lead_suit is None
    assert t.contains_points() is False


def test_game_phase_errors(game_4p):
    """Verify game rules enforcement in wrong phases."""
    # Not in passing phase
    with pytest.raises(ValueError):
        game_4p.set_pass_cards(0, [Card(Suit.HEARTS, Rank.TWO)])
    with pytest.raises(ValueError):
        game_4p.execute_pass()

    game_4p.start_round()  # PHASE -> PASSING
    # Wrong number of cards
    with pytest.raises(ValueError):
        game_4p.set_pass_cards(0, [Card(Suit.HEARTS, Rank.TWO)])  # Only 1 card, needs 3


def test_kitty_shuffle_logic():
    """Verify that KITTY mode re-shuffles if 2 of clubs would be at the bottom."""
    two_clubs = Card(Suit.CLUBS, Rank.TWO)

    shuffle_count = 0

    def mock_shuffle(d):
        nonlocal shuffle_count
        if shuffle_count == 0:
            d.remove(two_clubs)
            d.append(two_clubs)
        else:
            d.remove(two_clubs)
            d.insert(0, two_clubs)
        shuffle_count += 1

    with patch("random.shuffle", side_effect=mock_shuffle):
        game = HeartsGame(["P0", "P1", "P2"], three_player_mode=ThreePlayerMode.KITTY)
        game.start_round()
        assert shuffle_count >= 2
        assert game.kitty_card != two_clubs


def test_game_passing_edge_cases(game_3p_remove):
    """Verify passing errors and 3-player offsets."""
    game_3p_remove.start_round()
    # Player 0 does not have this card
    with pytest.raises(ValueError):
        game_3p_remove.set_pass_cards(
            0,
            [
                Card(Suit.SPADES, Rank.ACE),
                Card(Suit.SPADES, Rank.KING),
                Card(Suit.SPADES, Rank.QUEEN),
            ],
        )

    # Not all ready to pass
    with pytest.raises(ValueError):
        game_3p_remove.execute_pass()

    # Valid 3p pass execution (verify offsets)
    for i in range(3):
        hand = list(game_3p_remove.players[i].hand[:3])
        game_3p_remove.set_pass_cards(i, hand)
    game_3p_remove.execute_pass()
    assert game_3p_remove.phase == GamePhase.PLAYING


def test_play_card_errors(game_4p):
    """Verify play_card phase and turn errors."""
    # Phase error
    game_4p.phase = GamePhase.PASSING
    res = game_4p.play_card(0, Card(Suit.CLUBS, Rank.TWO))
    assert res["error"] == "Not in playing phase"

    # Turn error
    game_4p.phase = GamePhase.PLAYING
    game_4p.current_player_index = 1
    res = game_4p.play_card(0, Card(Suit.CLUBS, Rank.TWO))
    assert "Not player 0's turn" in res["error"]


def test_kitty_claim_and_realtime_scores():
    """Verify result dict when kitty is claimed and real-time score updates."""
    game = HeartsGame(["P0", "P1", "P2"], three_player_mode=ThreePlayerMode.KITTY)
    game.start_round()
    game.phase = GamePhase.PLAYING

    # Manually set hands to non-point cards except one Heart for p0
    h2 = Card(Suit.HEARTS, Rank.TWO)
    c3 = Card(Suit.CLUBS, Rank.THREE)
    c4 = Card(Suit.CLUBS, Rank.FOUR)

    game.players[0].hand = [h2]
    game.players[1].hand = [c3]
    game.players[2].hand = [c4]

    game.tricks_played = 1  # Bypass first trick 2-of-clubs rule
    game.current_player_index = 0
    game.current_trick = Trick(0, 3)

    # p0 leads Hearts 2 (points)
    res = game.play_card(0, h2)
    assert res["valid"], f"Play should be valid, error: {res.get('error')}"
    assert not res["trick_complete"]

    # p1 follows with Clubs 3 (void in hearts)
    res1 = game.play_card(1, c3)
    assert res1["valid"], f"p1 play should be valid, error: {res1.get('error')}"

    # p2 follows with Clubs 4 (void in hearts)
    res2 = game.play_card(2, c4)
    assert res2["valid"], f"p2 play should be valid, error: {res2.get('error')}"
    assert res2["trick_complete"]
    assert res2["trick_winner"] == 0
    assert res2["kitty_claimed"]

    # Verify real-time score update: p0 should have 1 point from Heart 2 + kitty points
    assert game.players[0].round_score > 0
