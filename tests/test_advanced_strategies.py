import pytest
from hearts_game import Card, Suit, Rank, Trick
from hearts_ai import HeartsAI, AIWeights


def test_bleeding_spades():
    """Verify AI leads spades when holding Queen and protection."""
    weights = AIWeights(bleed_spades_priority=50.0, queen_protection_threshold=3)
    ai = HeartsAI(weights=weights)

    # Hand with Q and 3 low spades (protected)
    hand = [
        Card(Suit.SPADES, Rank.QUEEN),
        Card(Suit.SPADES, Rank.TWO),
        Card(Suit.SPADES, Rank.THREE),
        Card(Suit.SPADES, Rank.FOUR),
        Card(Suit.CLUBS, Rank.FIVE),
    ]
    valid_plays = [
        Card(Suit.SPADES, Rank.TWO),
        Card(Suit.SPADES, Rank.THREE),
        Card(Suit.SPADES, Rank.FOUR),
        Card(Suit.CLUBS, Rank.FIVE),
    ]

    trick = Trick(lead_player_index=0)
    # AI is leading
    play = ai.select_play(hand, valid_plays, trick, 0, 4, False)

    # Should lead a low spade to "bleed" them
    assert play.suit == Suit.SPADES
    assert play.rank.value < Rank.QUEEN.value


def test_finesse_play():
    """Verify AI attempts a finesse (high duck) when not last and suit control is missing."""
    weights = AIWeights(high_duck_preference=2.0)
    ai = HeartsAI(weights=weights)

    # AI is 2nd to play
    # Lead was King. AI has 10 and J.
    # High club (A, K, Q) are still out there.
    hand = [Card(Suit.CLUBS, Rank.TEN), Card(Suit.CLUBS, Rank.JACK)]
    valid_plays = [Card(Suit.CLUBS, Rank.TEN), Card(Suit.CLUBS, Rank.JACK)]

    trick = Trick(lead_player_index=0)
    trick.add_card(0, Card(Suit.CLUBS, Rank.KING))

    # Mark A, K, Q as NOT played
    ai.tracker.reset()

    play = ai.select_play(hand, valid_plays, trick, 1, 4, False)

    # Should play the JACK (finesse/high duck) to draw out high cards from players 2 or 3
    assert play == Card(Suit.CLUBS, Rank.JACK)


def test_strategic_passing_2_of_clubs():
    """Verify AI considers passing 2 of clubs unless it has the Ace."""
    weights = AIWeights(
        pass_two_of_clubs_control=100.0, retain_ace_of_clubs_bonus=150.0
    )
    ai = HeartsAI(weights=weights)

    # Case 1: Has 2, but NO Ace. Should pass 2.
    hand = [
        Card(Suit.CLUBS, Rank.TWO),
        Card(Suit.DIAMONDS, Rank.ACE),
        Card(Suit.HEARTS, Rank.ACE),
        Card(Suit.SPADES, Rank.ACE),
    ]
    pass_cards = ai.select_pass_cards(hand, "LEFT")
    assert Card(Suit.CLUBS, Rank.TWO) in pass_cards

    # Case 2: Has 2 AND Ace. Should keep 2 to ensure it wins first trick and controls the lead.
    hand = [
        Card(Suit.CLUBS, Rank.TWO),
        Card(Suit.CLUBS, Rank.ACE),
        Card(Suit.DIAMONDS, Rank.KING),
        Card(Suit.HEARTS, Rank.KING),
    ]
    pass_cards = ai.select_pass_cards(hand, "LEFT")
    assert Card(Suit.CLUBS, Rank.TWO) not in pass_cards


def test_void_signaling_avoidance():
    """Verify AI avoids passing the last card of a suit if possible."""
    weights = AIWeights(pass_last_card_penalty=100.0)
    ai = HeartsAI(weights=weights)

    # Hand where Diamonds is a single card.
    hand = [
        Card(Suit.DIAMONDS, Rank.KING),  # Last card of suit
        Card(Suit.CLUBS, Rank.ACE),
        Card(Suit.CLUBS, Rank.KING),
        Card(Suit.CLUBS, Rank.QUEEN),
        Card(Suit.CLUBS, Rank.JACK),
    ]

    pass_cards = ai.select_pass_cards(hand, "LEFT")
    # Should NOT pass the Diamond King because it's the last card, even though it's high
    assert Card(Suit.DIAMONDS, Rank.KING) not in pass_cards
