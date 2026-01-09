import pytest
from unittest.mock import patch
from hearts_game import Card, Suit, Rank, GameMode, Trick
from hearts_ai import (
    HeartsAI,
    AIWeights,
    CardTracker,
    DEFAULT_WEIGHTS,
    set_ai_weights,
    get_ai_weights,
    get_ai,
    reset_all_ai,
    record_trick_for_all,
    get_weights_summary,
)


def test_ai_weight_selection():
    """Verify AI correctly picks mode-specific weights."""
    # Mock some weights
    w_4p = DEFAULT_WEIGHTS[GameMode.PLAYER_4]
    ai = HeartsAI(weights=w_4p)
    assert ai.weights == w_4p


def test_ai_passing_dangerous_cards():
    """Verify AI prioritizes passing dangerous cards (QoS, high cards)."""
    weights = AIWeights(pass_queen_of_spades=100.0, pass_high_cards=50.0)
    ai = HeartsAI(weights=weights)

    # Hand with QoS, Ace of Spades, and low clubs
    hand = [
        Card(Suit.SPADES, Rank.QUEEN),
        Card(Suit.SPADES, Rank.ACE),
        Card(Suit.CLUBS, Rank.TWO),
        Card(Suit.CLUBS, Rank.THREE),
        Card(Suit.CLUBS, Rank.FOUR),
        Card(Suit.CLUBS, Rank.FIVE),
        Card(Suit.CLUBS, Rank.SIX),
        Card(Suit.CLUBS, Rank.SEVEN),
        Card(Suit.CLUBS, Rank.EIGHT),
        Card(Suit.CLUBS, Rank.NINE),
        Card(Suit.CLUBS, Rank.TEN),
        Card(Suit.CLUBS, Rank.JACK),
        Card(Suit.CLUBS, Rank.KING),
    ]

    passed = ai.select_pass_cards(hand, "LEFT")
    assert len(passed) == 3
    assert Card(Suit.SPADES, Rank.QUEEN) in passed
    assert Card(Suit.SPADES, Rank.ACE) in passed


def test_ai_selection_complex():
    """Verify AI play selection in various positions."""
    ai = HeartsAI()
    ai.reset_round(num_players=4)

    hand = [Card(Suit.CLUBS, Rank.TWO), Card(Suit.CLUBS, Rank.ACE)]
    valid = [Card(Suit.CLUBS, Rank.TWO)]
    trick = Trick(lead_player_index=0)

    # Must lead 2 of clubs if we have it and it's first trick
    play = ai.select_play(hand, valid, trick, 0, 4, False)
    assert play == Card(Suit.CLUBS, Rank.TWO)

    # Test ducking: follow suit with lower card if possible
    hand = [Card(Suit.HEARTS, Rank.TEN), Card(Suit.HEARTS, Rank.THREE)]
    valid = hand
    trick = Trick(lead_player_index=0)
    trick.add_card(0, Card(Suit.HEARTS, Rank.FIVE))

    play = ai.select_play(hand, valid, trick, 1, 4, True)
    # Actually HeartsAI._select_follow: "Try to win with lowest card that beats lead if safe, or duck"
    # Ducking logic prefers lower cards to keep high cards for later if we must win.
    assert play == Card(Suit.HEARTS, Rank.THREE)


def test_void_tracking():
    """Verify CardTracker correctly records voids."""
    tracker = CardTracker()
    tracker.reset()

    # Trick: lead Clubs, but player 2 plays Spades
    trick_info = [
        (0, Card(Suit.CLUBS, Rank.TWO)),
        (1, Card(Suit.CLUBS, Rank.ACE)),
        (2, Card(Suit.SPADES, Rank.THREE)),  # Player 2 is void in Clubs
        (3, Card(Suit.CLUBS, Rank.KING)),
    ]

    # We need to pass the info to mark_player_void or use record_trick logic
    # In HeartsAI.record_trick:
    # if card.suit != trick_lead_suit: self.tracker.mark_player_void(p_idx, trick_lead_suit)

    lead_suit = Suit.CLUBS
    for p_idx, card in trick_info:
        if card.suit != lead_suit:
            tracker.mark_player_void(p_idx, lead_suit)

    assert tracker.is_player_void(2, Suit.CLUBS)
    assert not tracker.is_player_void(0, Suit.CLUBS)


def test_ai_shooting_moon_detection():
    """Verify AI detects shoot the moon threats."""
    weights = AIWeights(moon_threat_threshold=5, moon_block_priority=100.0)
    ai = HeartsAI(weights=weights)
    ai.reset_round(num_players=4)

    # Player 1 has taken 10 hearts
    trick_cards = [Card(Suit.HEARTS, Rank(r)) for r in range(2, 12)]
    ai.record_trick(trick_cards, winner_index=1)

    threat_index = ai._detect_moon_threat(my_index=0, num_players=4)
    assert threat_index == 1


def test_ai_shooting_moon_control():
    """Verify AI play selection when shooting the moon."""
    ai = HeartsAI()
    ai.reset_round(num_players=4)
    # Give AI a very strong hand
    hand = [
        Card(Suit.HEARTS, Rank.ACE),
        Card(Suit.HEARTS, Rank.KING),
        Card(Suit.SPADES, Rank.ACE),
    ]
    valid = hand
    trick = Trick(lead_player_index=0)

    # Lead: if shooting moon, lead high
    with patch.object(HeartsAI, "_should_attempt_moon", return_value=True):
        play = ai.select_play(hand, valid, trick, 0, 4, True)
        assert play.rank == Rank.ACE

    # Follow: if shooting moon, win with highest card
    trick.add_card(0, Card(Suit.HEARTS, Rank.TWO))
    with patch.object(HeartsAI, "_should_attempt_moon", return_value=True):
        play = ai.select_play(hand, valid, trick, 1, 4, True)
        assert play == Card(Suit.HEARTS, Rank.ACE)


def test_ai_moon_blocking():
    """Verify AI discards safely when another player is a moon threat."""
    weights = AIWeights(moon_threat_threshold=5)
    ai = HeartsAI(weights=weights)
    ai.reset_round(num_players=4)

    # Player 1 is a threat
    ai.record_trick([Card(Suit.HEARTS, Rank(r)) for r in range(2, 10)], winner_index=1)

    # Discard phase: we can't follow suit (Clubs)
    hand = [Card(Suit.HEARTS, Rank.TEN), Card(Suit.CLUBS, Rank.TWO)]
    valid = hand
    trick = Trick(lead_player_index=0)
    trick.add_card(0, Card(Suit.DIAMONDS, Rank.ACE))

    # Should discard a non-point card even if we have points
    # Actually _select_discard logic for moon_threat: "keep points (discard non-point cards)"
    # We are player 0, threat is player 1.
    play = ai.select_play(hand, valid, trick, 0, 4, True)
    # If the AI thinks it's shooting moon it might behave differently.
    # But it shouldn't here. Let's verify it picks the 2 of Clubs.
    assert play == Card(Suit.CLUBS, Rank.TWO)


def test_global_ai_management():
    """Verify global AI functions (set_ai_weights, reset_all_ai, etc)."""
    weights = AIWeights(pass_high_cards=123.0)
    set_ai_weights(weights)
    assert get_ai_weights() == weights

    reset_all_ai(num_players=4, mode=GameMode.PLAYER_4, weights=weights)
    ai0 = get_ai(0)
    assert ai0.weights == weights

    # Record trick for all
    cards = [Card(Suit.CLUBS, Rank.TWO), Card(Suit.CLUBS, Rank.THREE)]
    record_trick_for_all(cards, winner_index=0)
    assert ai0.tracker.is_played(cards[0])

    summary = get_weights_summary()
    assert "123.0" in summary
    assert "PLAYER_4" in summary


def test_ai_error_cases():
    """Verify AI edge cases."""
    ai = HeartsAI()
    with pytest.raises(ValueError):
        ai.select_play([], [], Trick(0), 0, 4, False)

    # Threshold 0
    ai.weights.moon_attempt_threshold = 0
    assert not ai._should_attempt_moon([])


def test_ai_global_defaults():
    """Verify AI global function defaults/fallbacks."""
    from hearts_ai import _ai_instances

    _ai_instances.clear()
    ai = get_ai(5)
    assert ai is not None

    # reset_all_ai using default weights (line 828)
    reset_all_ai(num_players=4)
    ai0 = get_ai(0)
    assert ai0.weights != AIWeights(pass_high_cards=123.0)  # Should be back to default


def test_card_tracker_exhaustive():
    """Verify all CardTracker methods."""
    tracker = CardTracker()
    tracker.reset()

    # Record some hearts
    hearts = [Card(Suit.HEARTS, Rank.TWO), Card(Suit.HEARTS, Rank.THREE)]
    tracker.record_cards(hearts)
    assert tracker.hearts_played_count() == 2
    assert tracker.is_played(hearts[0])
    assert not tracker.queen_of_spades_played()

    # Check remaining in suit
    rem = tracker.remaining_in_suit(Suit.HEARTS)
    assert len(rem) == 11  # 13 - 2
    assert Card(Suit.HEARTS, Rank.FOUR) in rem

    # Check remaining with exclusion
    rem_ex = tracker.remaining_in_suit(
        Suit.HEARTS, excluding_hand=[Card(Suit.HEARTS, Rank.FOUR)]
    )
    assert len(rem_ex) == 10
    assert Card(Suit.HEARTS, Rank.FOUR) not in rem_ex
