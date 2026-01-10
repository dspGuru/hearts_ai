"""
Tests to improve code coverage for hearts_ai.py and hearts_game.py.
Targets specific uncovered lines identified in coverage analysis.
"""

import pytest
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
from hearts_ai import (
    HeartsAI,
    AIWeights,
    CardTracker,
    reset_all_ai,
    get_ai,
    record_trick_for_all,
)


# =============================================================================
# CardTracker Coverage Tests
# =============================================================================


class TestCardTrackerCoverage:
    """Tests for CardTracker methods not fully covered."""

    def test_record_passed_cards(self):
        """Cover CardTracker.record_passed_cards (line 269)."""
        tracker = CardTracker()
        cards = [Card(Suit.HEARTS, Rank.ACE), Card(Suit.SPADES, Rank.KING)]
        tracker.record_passed_cards(destination_player=2, cards=cards)
        assert tracker.passed_cards[2] == cards
        assert len(tracker.passed_cards[2]) == 2

    def test_get_high_card_probabilities_basic(self):
        """Cover get_high_card_probabilities basic path (lines 316-325)."""
        tracker = CardTracker()
        # No cards played yet
        probs = tracker.get_high_card_probabilities(Suit.HEARTS, num_players=4)
        # All players have equal probability
        assert len(probs) == 4
        assert all(p == 0.25 for p in probs.values())

    def test_get_high_card_probabilities_no_remaining(self):
        """Cover when no cards remain in suit (line 318-319)."""
        tracker = CardTracker()
        # Play all hearts
        for rank in Rank:
            tracker.record_cards([Card(Suit.HEARTS, rank)])
        probs = tracker.get_high_card_probabilities(Suit.HEARTS, num_players=4)
        assert all(p == 0.0 for p in probs.values())

    def test_get_high_card_probabilities_no_high_remaining(self):
        """Cover when no high cards remain (lines 324-325)."""
        tracker = CardTracker()
        # Play all high hearts (A, K, Q, J)
        tracker.record_cards([Card(Suit.HEARTS, Rank.ACE)])
        tracker.record_cards([Card(Suit.HEARTS, Rank.KING)])
        tracker.record_cards([Card(Suit.HEARTS, Rank.QUEEN)])
        tracker.record_cards([Card(Suit.HEARTS, Rank.JACK)])
        probs = tracker.get_high_card_probabilities(Suit.HEARTS, num_players=4)
        assert all(p == 0.0 for p in probs.values())

    def test_get_high_card_probabilities_with_voids(self):
        """Cover probability with void players (lines 328-336)."""
        tracker = CardTracker()
        # Mark players 1 and 2 as void in hearts
        tracker.mark_player_void(1, Suit.HEARTS)
        tracker.mark_player_void(2, Suit.HEARTS)
        probs = tracker.get_high_card_probabilities(Suit.HEARTS, num_players=4)
        # Only players 0 and 3 can have hearts
        assert probs[0] == 0.5
        assert probs[1] == 0.0
        assert probs[2] == 0.0
        assert probs[3] == 0.5

    def test_get_high_card_probabilities_all_void(self):
        """Cover when all players are void (lines 331-332)."""
        tracker = CardTracker()
        for i in range(4):
            tracker.mark_player_void(i, Suit.HEARTS)
        probs = tracker.get_high_card_probabilities(Suit.HEARTS, num_players=4)
        assert all(p == 0.0 for p in probs.values())

    def test_get_high_card_probabilities_with_passed_cards(self):
        """Cover probability adjustment from passed cards (lines 339-349)."""
        tracker = CardTracker()
        # Record that we passed the Ace of Hearts to player 2
        tracker.record_passed_cards(2, [Card(Suit.HEARTS, Rank.ACE)])
        probs = tracker.get_high_card_probabilities(Suit.HEARTS, num_players=4)
        # Player 2 should have boosted probability
        assert probs[2] > probs[0]


# =============================================================================
# HeartsAI Coverage Tests
# =============================================================================


class TestHeartsAIRecordTrick:
    """Tests for record_trick with trick_info (lines 395-398)."""

    def test_record_trick_marks_voids(self):
        """Cover void tracking via trick_info."""
        ai = HeartsAI()
        ai.reset_round(4)

        # Trick where player 2 doesn't follow suit (void in clubs)
        trick_info = [
            (0, Card(Suit.CLUBS, Rank.TWO)),
            (1, Card(Suit.CLUBS, Rank.ACE)),
            (2, Card(Suit.HEARTS, Rank.THREE)),  # Void in clubs!
            (3, Card(Suit.CLUBS, Rank.KING)),
        ]
        trick_cards = [card for _, card in trick_info]

        ai.record_trick(trick_cards, winner_index=1, trick_info=trick_info)

        assert ai.tracker.is_player_void(2, Suit.CLUBS)
        assert not ai.tracker.is_player_void(0, Suit.CLUBS)
        assert not ai.tracker.is_player_void(1, Suit.CLUBS)
        assert not ai.tracker.is_player_void(3, Suit.CLUBS)


class TestPassingDirectionBonuses:
    """Tests for passing direction bonuses (lines 447, 476, 487)."""

    def test_pass_across_bonus(self):
        """Cover ACROSS direction bonus (line 447, 487)."""
        weights = AIWeights(
            pass_queen_of_spades=100.0,
            pass_across_danger_bonus=50.0,
            queen_protection_threshold=5,  # Make QoS unprotected
            pass_high_hearts=70.0,
        )
        ai = HeartsAI(weights=weights)

        # Hand with unprotected QoS and high hearts
        hand = [
            Card(Suit.SPADES, Rank.QUEEN),
            Card(Suit.SPADES, Rank.TWO),  # Only 1 low spade - unprotected
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.HEARTS, Rank.KING),
            Card(Suit.CLUBS, Rank.TWO),
            Card(Suit.CLUBS, Rank.THREE),
            Card(Suit.CLUBS, Rank.FOUR),
            Card(Suit.CLUBS, Rank.FIVE),
            Card(Suit.CLUBS, Rank.SIX),
            Card(Suit.CLUBS, Rank.SEVEN),
            Card(Suit.CLUBS, Rank.EIGHT),
            Card(Suit.CLUBS, Rank.NINE),
            Card(Suit.CLUBS, Rank.TEN),
        ]

        passed = ai.select_pass_cards(hand, "ACROSS")
        # QoS should be passed with ACROSS bonus
        assert Card(Suit.SPADES, Rank.QUEEN) in passed

    def test_pass_right_void_bonus(self):
        """Cover RIGHT direction void bonus (line 476)."""
        weights = AIWeights(
            pass_to_void_suit=80.0,
            pass_right_void_bonus=30.0,
        )
        ai = HeartsAI(weights=weights)

        # Hand with a short suit (3 diamonds)
        hand = [
            Card(Suit.DIAMONDS, Rank.TWO),
            Card(Suit.DIAMONDS, Rank.THREE),
            Card(Suit.DIAMONDS, Rank.FOUR),
            Card(Suit.CLUBS, Rank.FIVE),
            Card(Suit.CLUBS, Rank.SIX),
            Card(Suit.CLUBS, Rank.SEVEN),
            Card(Suit.CLUBS, Rank.EIGHT),
            Card(Suit.CLUBS, Rank.NINE),
            Card(Suit.CLUBS, Rank.TEN),
            Card(Suit.CLUBS, Rank.JACK),
            Card(Suit.CLUBS, Rank.QUEEN),
            Card(Suit.CLUBS, Rank.KING),
            Card(Suit.CLUBS, Rank.ACE),
        ]

        passed = ai.select_pass_cards(hand, "RIGHT")
        # Should void diamonds with RIGHT bonus
        assert (
            Card(Suit.DIAMONDS, Rank.TWO) in passed
            or Card(Suit.DIAMONDS, Rank.THREE) in passed
            or Card(Suit.DIAMONDS, Rank.FOUR) in passed
        )


class TestMoonThreatLead:
    """Tests for moon threat leading hearts (lines 687-688)."""

    def test_lead_hearts_when_moon_threat(self):
        """Cover leading hearts to block moon shooter."""
        weights = AIWeights(moon_threat_threshold=5)
        ai = HeartsAI(weights=weights)
        ai.reset_round(4)

        # Player 1 has taken many points (moon threat)
        ai.points_taken[1] = 15
        ai.points_taken[0] = 0
        ai.points_taken[2] = 0
        ai.points_taken[3] = 0

        hand = [
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.HEARTS, Rank.TWO),
            Card(Suit.CLUBS, Rank.THREE),
        ]
        valid_plays = hand.copy()
        trick = Trick(lead_player_index=0)

        play = ai.select_play(hand, valid_plays, trick, 0, 4, True)

        # Should lead high heart to spread points
        assert play == Card(Suit.HEARTS, Rank.ACE)


class TestQueenFlushingStrategy:
    """Tests for queen flushing strategy (lines 723-725)."""

    def test_flush_queen_with_low_spades(self):
        """Cover leading low spades to flush QoS."""
        weights = AIWeights(
            flush_queen_priority=50.0,
            flush_queen_max_rank=8,
            flush_queen_min_spades=2,
            lead_spades_priority=0,  # Normally avoid leading spades
        )
        ai = HeartsAI(weights=weights)
        ai.reset_round(4)

        # QoS not played, we don't have it, have low spades
        hand = [
            Card(Suit.SPADES, Rank.TWO),
            Card(Suit.SPADES, Rank.THREE),
            Card(Suit.CLUBS, Rank.FOUR),
        ]
        valid_plays = hand.copy()
        trick = Trick(lead_player_index=0)

        play = ai.select_play(hand, valid_plays, trick, 0, 4, False)

        # Should lead low spade to flush queen
        assert play.suit == Suit.SPADES
        assert play.rank.value <= 8


class TestAvoidVoidOpponent:
    """Tests for avoiding leading when opponent is void (lines 734-739)."""

    def test_avoid_leading_when_opponent_void(self):
        """Cover lead_avoid_void_opponent penalty."""
        weights = AIWeights(
            lead_avoid_void_opponent=100.0,
            lead_clubs_priority=40.0,
            lead_diamonds_priority=30.0,
        )
        ai = HeartsAI(weights=weights)
        ai.reset_round(4)

        # Mark player 1 as void in clubs
        ai.tracker.mark_player_void(1, Suit.CLUBS)

        hand = [
            Card(Suit.CLUBS, Rank.TWO),
            Card(Suit.DIAMONDS, Rank.THREE),
        ]
        valid_plays = hand.copy()
        trick = Trick(lead_player_index=0)

        play = ai.select_play(hand, valid_plays, trick, 0, 4, False)

        # Should avoid clubs because player 1 is void (can dump points)
        assert play == Card(Suit.DIAMONDS, Rank.THREE)


class TestMoonBlockingFollow:
    """Tests for moon blocking when following (lines 815-817)."""

    def test_win_cheaply_to_block_moon(self):
        """Cover winning cheaply to block moon shooter."""
        weights = AIWeights(moon_threat_threshold=5, low_win_preference=1.0)
        ai = HeartsAI(weights=weights)
        ai.reset_round(4)

        # Player 2 is shooting moon
        ai.points_taken[2] = 15
        ai.points_taken[0] = 0
        ai.points_taken[1] = 0
        ai.points_taken[3] = 0

        # Trick has points (hearts)
        trick = Trick(lead_player_index=2, num_players=4)
        trick.add_card(2, Card(Suit.HEARTS, Rank.TWO))
        trick.add_card(3, Card(Suit.HEARTS, Rank.THREE))

        hand = [
            Card(Suit.HEARTS, Rank.KING),
            Card(Suit.HEARTS, Rank.FIVE),
        ]
        valid_plays = hand.copy()

        play = ai.select_play(hand, valid_plays, trick, 0, 4, True)

        # Should win cheaply (5) to block, not play King
        assert play == Card(Suit.HEARTS, Rank.FIVE)


class TestLastPositionPlay:
    """Tests for last position play decisions (lines 829-845)."""

    def test_last_no_points_take_trick(self):
        """Cover taking safe trick when last and no points."""
        weights = AIWeights(take_safe_trick_preference=1.0)
        ai = HeartsAI(weights=weights)
        ai.reset_round(4)

        # Trick with no points
        trick = Trick(lead_player_index=0, num_players=4)
        trick.add_card(0, Card(Suit.CLUBS, Rank.TWO))
        trick.add_card(1, Card(Suit.CLUBS, Rank.THREE))
        trick.add_card(2, Card(Suit.CLUBS, Rank.FOUR))

        hand = [
            Card(Suit.CLUBS, Rank.ACE),
            Card(Suit.CLUBS, Rank.FIVE),
        ]
        valid_plays = hand.copy()

        play = ai.select_play(hand, valid_plays, trick, 3, 4, False)

        # Last to play, no points - take trick with highest
        assert play == Card(Suit.CLUBS, Rank.ACE)

    def test_last_with_points_duck(self):
        """Cover ducking when last and trick has points."""
        weights = AIWeights(high_duck_preference=1.0)
        ai = HeartsAI(weights=weights)
        ai.reset_round(4)

        # Trick with points (heart)
        trick = Trick(lead_player_index=0, num_players=4)
        trick.add_card(0, Card(Suit.CLUBS, Rank.TEN))
        trick.add_card(1, Card(Suit.HEARTS, Rank.TWO))  # Point card
        trick.add_card(2, Card(Suit.CLUBS, Rank.FOUR))

        hand = [
            Card(Suit.CLUBS, Rank.ACE),
            Card(Suit.CLUBS, Rank.FIVE),
        ]
        valid_plays = hand.copy()

        play = ai.select_play(hand, valid_plays, trick, 3, 4, True)

        # Last to play, has points - duck with highest ducker (5)
        assert play == Card(Suit.CLUBS, Rank.FIVE)

    def test_last_with_points_must_win(self):
        """Cover must-win scenario when last with points."""
        weights = AIWeights(low_win_preference=1.0)
        ai = HeartsAI(weights=weights)
        ai.reset_round(4)

        # Trick with points where we must win
        trick = Trick(lead_player_index=0, num_players=4)
        trick.add_card(0, Card(Suit.CLUBS, Rank.TEN))
        trick.add_card(1, Card(Suit.HEARTS, Rank.TWO))  # Point card
        trick.add_card(2, Card(Suit.CLUBS, Rank.JACK))

        # We only have cards higher than Jack
        hand = [
            Card(Suit.CLUBS, Rank.ACE),
            Card(Suit.CLUBS, Rank.KING),
        ]
        valid_plays = hand.copy()

        play = ai.select_play(hand, valid_plays, trick, 3, 4, True)

        # Must win - play lowest winner
        assert play == Card(Suit.CLUBS, Rank.KING)


class TestExitCardPreservation:
    """Tests for exit card preservation (lines 858-870)."""

    def test_preserve_exit_cards_when_must_win(self):
        """Cover exit card preservation when not last and must win."""
        weights = AIWeights(
            exit_card_threshold=6,
            low_win_preference=1.0,
        )
        ai = HeartsAI(weights=weights)
        ai.reset_round(4)

        # Not last, must win (no ducking options)
        trick = Trick(lead_player_index=0, num_players=4)
        trick.add_card(0, Card(Suit.CLUBS, Rank.JACK))  # High lead, forces win

        # All our cards beat the Jack, so we must win
        # We have non-exit winners only
        hand = [
            Card(Suit.CLUBS, Rank.KING),  # Non-exit winner
            Card(Suit.CLUBS, Rank.ACE),  # Non-exit winner
            Card(Suit.CLUBS, Rank.QUEEN),  # Non-exit winner
        ]
        valid_plays = hand.copy()

        play = ai.select_play(hand, valid_plays, trick, 1, 4, False)

        # Should use lowest non-exit winner (Queen)
        assert play == Card(Suit.CLUBS, Rank.QUEEN)

    def test_must_use_exit_card_if_only_winners(self):
        """Cover when only exit cards can win."""
        weights = AIWeights(
            exit_card_threshold=6,
            low_win_preference=1.0,
        )
        ai = HeartsAI(weights=weights)
        ai.reset_round(4)

        trick = Trick(lead_player_index=0, num_players=4)
        trick.add_card(0, Card(Suit.CLUBS, Rank.TWO))

        # Only have exit-level cards that win
        hand = [
            Card(Suit.CLUBS, Rank.THREE),
            Card(Suit.CLUBS, Rank.FOUR),
        ]
        valid_plays = hand.copy()

        play = ai.select_play(hand, valid_plays, trick, 1, 4, False)

        # Must play lowest winner even if it's an exit card
        assert play == Card(Suit.CLUBS, Rank.THREE)


class TestMoonShootingDiscard:
    """Tests for moon shooting discard strategy (lines 891-903)."""

    def test_moon_shooting_discard_with_suit_control(self):
        """Cover discarding low hearts when shooting moon with suit control."""
        weights = AIWeights(
            control_card_count=2,
            moon_attempt_threshold=50,
            moon_high_hearts_weight=10,
            moon_queen_spades_weight=15,
        )
        ai = HeartsAI(weights=weights)
        ai.reset_round(4)

        # Strong moon hand with heart control
        hand = [
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.HEARTS, Rank.KING),
            Card(Suit.HEARTS, Rank.QUEEN),
            Card(Suit.HEARTS, Rank.JACK),
            Card(Suit.HEARTS, Rank.TWO),  # Low heart to potentially dump
            Card(Suit.SPADES, Rank.QUEEN),
            Card(Suit.CLUBS, Rank.THREE),  # Non-heart, non-point
        ]

        # Can't follow suit (discarding)
        trick = Trick(lead_player_index=0, num_players=4)
        trick.add_card(0, Card(Suit.DIAMONDS, Rank.ACE))

        valid_plays = [
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.HEARTS, Rank.KING),
            Card(Suit.HEARTS, Rank.QUEEN),
            Card(Suit.HEARTS, Rank.JACK),
            Card(Suit.HEARTS, Rank.TWO),
            Card(Suit.SPADES, Rank.QUEEN),
            Card(Suit.CLUBS, Rank.THREE),
        ]

        play = ai.select_play(hand, valid_plays, trick, 1, 4, True)

        # With heart control, should dump low heart
        assert play == Card(Suit.HEARTS, Rank.TWO)

    def test_moon_shooting_discard_safe(self):
        """Cover safe discard when NOT shooting moon - standard discard."""
        # Use low thresholds so AI doesn't think it's shooting moon
        weights = AIWeights(
            moon_attempt_threshold=100,  # Very high threshold = won't attempt moon
        )
        ai = HeartsAI(weights=weights)
        ai.reset_round(4)

        # Weak hand - not shooting moon
        hand = [
            Card(Suit.CLUBS, Rank.THREE),
            Card(Suit.CLUBS, Rank.KING),
            Card(Suit.DIAMONDS, Rank.FIVE),
        ]

        trick = Trick(lead_player_index=0, num_players=4)
        trick.add_card(0, Card(Suit.SPADES, Rank.ACE))  # Lead spades

        valid_plays = hand.copy()

        play = ai.select_play(hand, valid_plays, trick, 1, 4, False)

        # Standard discard: should discard highest card
        assert play == Card(Suit.CLUBS, Rank.KING)


class TestStandardDiscard:
    """Tests for standard discard logic (lines 912-949)."""

    def test_discard_queen_of_spades_highest_priority(self):
        """Cover QoS as highest discard priority."""
        ai = HeartsAI()
        ai.reset_round(4)

        hand = [
            Card(Suit.SPADES, Rank.QUEEN),
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.CLUBS, Rank.TWO),
        ]

        trick = Trick(lead_player_index=0, num_players=4)
        trick.add_card(0, Card(Suit.DIAMONDS, Rank.ACE))

        valid_plays = hand.copy()

        play = ai.select_play(hand, valid_plays, trick, 1, 4, True)

        assert play == Card(Suit.SPADES, Rank.QUEEN)

    def test_discard_hearts_second_priority(self):
        """Cover hearts discard priority."""
        ai = HeartsAI()
        ai.reset_round(4)

        hand = [
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.CLUBS, Rank.TWO),
        ]

        trick = Trick(lead_player_index=0, num_players=4)
        trick.add_card(0, Card(Suit.DIAMONDS, Rank.ACE))

        valid_plays = hand.copy()

        play = ai.select_play(hand, valid_plays, trick, 1, 4, True)

        assert play == Card(Suit.HEARTS, Rank.ACE)

    def test_discard_dangerous_spades(self):
        """Cover dangerous spades (A/K) discard when QoS not played."""
        ai = HeartsAI()
        ai.reset_round(4)
        # QoS not played yet

        hand = [
            Card(Suit.SPADES, Rank.ACE),
            Card(Suit.CLUBS, Rank.TWO),
        ]

        trick = Trick(lead_player_index=0, num_players=4)
        trick.add_card(0, Card(Suit.DIAMONDS, Rank.ACE))

        valid_plays = hand.copy()

        play = ai.select_play(hand, valid_plays, trick, 1, 4, False)

        # Should discard dangerous Ace of Spades
        assert play == Card(Suit.SPADES, Rank.ACE)

    def test_discard_high_cards_fallback(self):
        """Cover high card discard fallback."""
        ai = HeartsAI()
        ai.reset_round(4)
        # Mark QoS as played
        ai.tracker.record_cards([Card(Suit.SPADES, Rank.QUEEN)])

        hand = [
            Card(Suit.CLUBS, Rank.ACE),
            Card(Suit.CLUBS, Rank.TWO),
        ]

        trick = Trick(lead_player_index=0, num_players=4)
        trick.add_card(0, Card(Suit.DIAMONDS, Rank.ACE))

        valid_plays = hand.copy()

        play = ai.select_play(hand, valid_plays, trick, 1, 4, False)

        # Should discard high card
        assert play == Card(Suit.CLUBS, Rank.ACE)


class TestHasSuitControl:
    """Tests for _has_suit_control (lines 953-957)."""

    def test_has_suit_control_true(self):
        """Cover having suit control."""
        weights = AIWeights(control_card_count=2)
        ai = HeartsAI(weights=weights)

        hand = [
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.HEARTS, Rank.KING),
            Card(Suit.HEARTS, Rank.TWO),
        ]

        assert ai._has_suit_control(hand, Suit.HEARTS)

    def test_has_suit_control_false(self):
        """Cover not having suit control."""
        weights = AIWeights(control_card_count=3)
        ai = HeartsAI(weights=weights)

        hand = [
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.HEARTS, Rank.TWO),
        ]

        assert not ai._has_suit_control(hand, Suit.HEARTS)


# =============================================================================
# HeartsGame Coverage Tests
# =============================================================================


class TestTrickWinner:
    """Tests for Trick.get_winner edge cases (lines 230-231)."""

    def test_trick_winner_later_card_wins(self):
        """Cover winning card that's not the first card."""
        trick = Trick(lead_player_index=0, num_players=4)
        trick.add_card(0, Card(Suit.CLUBS, Rank.TWO))
        trick.add_card(1, Card(Suit.CLUBS, Rank.ACE))
        trick.add_card(2, Card(Suit.CLUBS, Rank.THREE))
        trick.add_card(3, Card(Suit.CLUBS, Rank.FOUR))

        assert trick.get_winner() == 1

    def test_trick_winner_off_suit_doesnt_win(self):
        """Cover off-suit card not winning even if higher."""
        trick = Trick(lead_player_index=0, num_players=4)
        trick.add_card(0, Card(Suit.CLUBS, Rank.TWO))
        trick.add_card(1, Card(Suit.HEARTS, Rank.ACE))  # Off-suit, high
        trick.add_card(2, Card(Suit.CLUBS, Rank.THREE))
        trick.add_card(3, Card(Suit.CLUBS, Rank.FOUR))

        # Player 3 wins with highest club, not player 1 with Ace of hearts
        assert trick.get_winner() == 3


class TestNoPassRound:
    """Tests for no-pass round (lines 377-378)."""

    def test_no_pass_direction_starts_playing(self):
        """Cover PassDirection.NONE going straight to playing."""
        game = HeartsGame(["P0", "P1", "P2", "P3"])
        # Round 4 is NONE for 4-player
        game.round_number = 3  # Will become 4 in start_round
        game.start_round()

        # Should skip passing phase
        assert game.phase == GamePhase.PLAYING
        assert game.pass_direction == PassDirection.NONE


class TestFourPlayerPassingOffsets:
    """Tests for 4-player passing offsets (lines 410-416)."""

    def test_pass_left_offset(self):
        """Cover passing left (offset 1)."""
        game = HeartsGame(["P0", "P1", "P2", "P3"])
        game.start_round()  # Round 1 = LEFT
        assert game.pass_direction == PassDirection.LEFT

        # Set up passes
        for i in range(4):
            cards = list(game.players[i].hand[:3])
            game.set_pass_cards(i, cards)

        p0_original = list(game.players[0].hand[:3])
        game.execute_pass()

        # P0's cards should go to P1 (left)
        # P3's cards should come to P0
        # Just verify the pass happened
        assert game.phase == GamePhase.PLAYING

    def test_pass_right_offset(self):
        """Cover passing right (offset 3 for 4p)."""
        game = HeartsGame(["P0", "P1", "P2", "P3"])
        game.round_number = 1  # Round 2 = RIGHT
        game.start_round()
        assert game.pass_direction == PassDirection.RIGHT

        for i in range(4):
            cards = list(game.players[i].hand[:3])
            game.set_pass_cards(i, cards)

        game.execute_pass()
        assert game.phase == GamePhase.PLAYING

    def test_pass_across_offset(self):
        """Cover passing across (offset 2 for 4p)."""
        game = HeartsGame(["P0", "P1", "P2", "P3"])
        game.round_number = 2  # Round 3 = ACROSS
        game.start_round()
        assert game.pass_direction == PassDirection.ACROSS

        for i in range(4):
            cards = list(game.players[i].hand[:3])
            game.set_pass_cards(i, cards)

        game.execute_pass()
        assert game.phase == GamePhase.PLAYING


class TestFirstTrickPointsRestriction:
    """Tests for first trick points restriction (lines 478, 483-485)."""

    def test_cannot_dump_points_first_trick(self):
        """Cover restriction on dumping points on first trick."""
        game = HeartsGame(["P0", "P1", "P2", "P3"])
        game.start_round()
        game.phase = GamePhase.PLAYING
        game._start_first_trick()

        # Find player with 2 of clubs
        lead_idx = game.current_player_index

        # Play 2 of clubs
        game.play_card(lead_idx, Card(Suit.CLUBS, Rank.TWO))

        # Next player's valid plays
        next_idx = game.current_player_index
        player = game.players[next_idx]

        # Give player only hearts and QoS (no clubs)
        player.hand = [
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.HEARTS, Rank.KING),
            Card(Suit.SPADES, Rank.QUEEN),
            Card(Suit.DIAMONDS, Rank.TWO),
        ]

        valid = game.get_valid_plays(next_idx)

        # Should only allow non-point cards on first trick
        assert Card(Suit.HEARTS, Rank.ACE) not in valid
        assert Card(Suit.SPADES, Rank.QUEEN) not in valid
        assert Card(Suit.DIAMONDS, Rank.TWO) in valid

    def test_must_play_points_if_only_option(self):
        """Cover playing points when no other choice on first trick."""
        game = HeartsGame(["P0", "P1", "P2", "P3"])
        game.start_round()
        game.phase = GamePhase.PLAYING
        game._start_first_trick()

        lead_idx = game.current_player_index
        game.play_card(lead_idx, Card(Suit.CLUBS, Rank.TWO))

        next_idx = game.current_player_index
        player = game.players[next_idx]

        # Give player ONLY hearts (no other choice)
        player.hand = [
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.HEARTS, Rank.KING),
        ]

        valid = game.get_valid_plays(next_idx)

        # Must allow hearts since that's all they have
        assert len(valid) == 2
        assert Card(Suit.HEARTS, Rank.ACE) in valid


class TestLeadingHeartsOnly:
    """Test leading when only hearts remain (line 494)."""

    def test_lead_hearts_when_only_hearts(self):
        """Cover leading hearts when that's all player has."""
        game = HeartsGame(["P0", "P1", "P2", "P3"])
        game.start_round()
        game.phase = GamePhase.PLAYING
        game.hearts_broken = False
        game.tricks_played = 1  # Not first trick

        # Player 0 leads
        game.current_player_index = 0
        game.current_trick = Trick(lead_player_index=0, num_players=4)

        # Player 0 only has hearts
        game.players[0].hand = [
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.HEARTS, Rank.TWO),
        ]

        valid = game.get_valid_plays(0)

        # Should allow leading hearts (breaks hearts)
        assert len(valid) == 2


class TestRoundCompletion:
    """Tests for round completion (lines 586-593)."""

    def test_full_round_completion(self):
        """Cover complete round flow including round_complete flag."""
        game = HeartsGame(["P0", "P1", "P2", "P3"])
        game.start_round()
        game.phase = GamePhase.PLAYING
        game._start_first_trick()

        # Play through an entire round
        for trick_num in range(13):
            for _ in range(4):
                p_idx = game.current_player_index
                valid = game.get_valid_plays(p_idx)
                if valid:
                    result = game.play_card(p_idx, valid[0])
                    if result["round_complete"]:
                        assert game.phase == GamePhase.ROUND_END
                        return

        # If we get here, round should be complete
        assert game.phase == GamePhase.ROUND_END


class TestGetWinner:
    """Tests for get_winner function (lines 635, 639-641)."""

    def test_get_winner_game_not_over(self):
        """Cover get_winner when game not over."""
        game = HeartsGame(["P0", "P1", "P2", "P3"])
        assert game.get_winner() is None

    def test_get_winner_game_over(self):
        """Cover get_winner when game is over."""
        game = HeartsGame(["P0", "P1", "P2", "P3"])
        game.players[0].total_score = 100
        game.players[1].total_score = 50
        game.players[2].total_score = 30
        game.players[3].total_score = 20

        winner = game.get_winner()
        assert winner == game.players[3]
        assert winner.name == "P3"


class TestTricksPerRound:
    """Tests for tricks_per_round property (line 653)."""

    def test_tricks_per_round_4p(self):
        """Cover 4-player tricks per round."""
        game = HeartsGame(["P0", "P1", "P2", "P3"])
        assert game.tricks_per_round == 13

    def test_tricks_per_round_3p(self):
        """Cover 3-player tricks per round."""
        game = HeartsGame(["P0", "P1", "P2"])
        assert game.tricks_per_round == 17


class TestKittyStateInGameState:
    """Tests for kitty info in get_game_state (lines 685-686)."""

    def test_game_state_includes_kitty_info(self):
        """Cover kitty card in game state for 3p kitty mode."""
        game = HeartsGame(["P0", "P1", "P2"], three_player_mode=ThreePlayerMode.KITTY)
        game.start_round()

        state = game.get_game_state()

        assert "kitty_claimed" in state
        assert "kitty_card" in state
        assert state["kitty_claimed"] is False
        assert state["kitty_card"] is not None


class TestAllPlayersReadyToPass:
    """Test all_players_ready_to_pass edge cases."""

    def test_all_ready_returns_true(self):
        """Cover when all players have set pass cards."""
        game = HeartsGame(["P0", "P1", "P2", "P3"])
        game.start_round()

        for i in range(4):
            cards = list(game.players[i].hand[:3])
            game.set_pass_cards(i, cards)

        assert game.all_players_ready_to_pass()

    def test_not_all_ready_returns_false(self):
        """Cover when not all players have set pass cards."""
        game = HeartsGame(["P0", "P1", "P2", "P3"])
        game.start_round()

        # Only player 0 sets cards
        cards = list(game.players[0].hand[:3])
        game.set_pass_cards(0, cards)

        assert not game.all_players_ready_to_pass()


class TestGetCurrentPlayer:
    """Test get_current_player method."""

    def test_get_current_player(self):
        """Cover get_current_player method (line 653)."""
        game = HeartsGame(["P0", "P1", "P2", "P3"])
        game.current_player_index = 2

        current = game.get_current_player()
        assert current == game.players[2]
        assert current.name == "P2"


class TestValidPlaysEdgeCases:
    """Tests for get_valid_plays edge cases (lines 458, 460, 466)."""

    def test_valid_plays_wrong_phase(self):
        """Cover returning empty list when not in playing phase (line 458)."""
        game = HeartsGame(["P0", "P1", "P2", "P3"])
        game.start_round()
        # Phase is PASSING, not PLAYING
        assert game.phase == GamePhase.PASSING

        valid = game.get_valid_plays(0)
        assert valid == []

    def test_valid_plays_wrong_player(self):
        """Cover returning empty list when not player's turn (line 460)."""
        game = HeartsGame(["P0", "P1", "P2", "P3"])
        game.start_round()
        game.phase = GamePhase.PLAYING
        game._start_first_trick()

        # Current player is whoever has 2 of clubs
        current = game.current_player_index
        wrong_player = (current + 1) % 4

        valid = game.get_valid_plays(wrong_player)
        assert valid == []

    def test_valid_plays_empty_hand(self):
        """Cover returning empty list when hand is empty (line 466)."""
        game = HeartsGame(["P0", "P1", "P2", "P3"])
        game.start_round()
        game.phase = GamePhase.PLAYING
        game.current_player_index = 0
        game.current_trick = Trick(lead_player_index=0, num_players=4)

        # Empty the player's hand
        game.players[0].hand = []

        valid = game.get_valid_plays(0)
        assert valid == []


class TestGameOverTransition:
    """Tests for game over state transition (lines 590-591)."""

    def test_round_ends_with_game_over(self):
        """Cover game over phase transition."""
        game = HeartsGame(["P0", "P1", "P2", "P3"])
        game.start_round()
        game.phase = GamePhase.PLAYING
        game._start_first_trick()

        # Set player 0's score near the limit
        game.players[0].total_score = 95

        # Play a complete round (simplified - just set up end state)
        game.tricks_played = 12  # One trick away from end

        # Set up last trick
        game.current_trick = Trick(lead_player_index=0, num_players=4)

        # Give each player one card
        game.players[0].hand = [Card(Suit.CLUBS, Rank.TWO)]
        game.players[1].hand = [Card(Suit.HEARTS, Rank.ACE)]
        game.players[2].hand = [Card(Suit.HEARTS, Rank.KING)]
        game.players[3].hand = [Card(Suit.HEARTS, Rank.QUEEN)]

        game.current_player_index = 0

        # Play through the final trick
        result = game.play_card(0, Card(Suit.CLUBS, Rank.TWO))
        assert result["valid"]

        result = game.play_card(1, Card(Suit.HEARTS, Rank.ACE))
        assert result["valid"]

        result = game.play_card(2, Card(Suit.HEARTS, Rank.KING))
        assert result["valid"]

        result = game.play_card(3, Card(Suit.HEARTS, Rank.QUEEN))
        assert result["valid"]
        assert result["trick_complete"]
        assert result["round_complete"]

        # Check if game transitioned to GAME_OVER or ROUND_END
        # Since player 0 has 95 + points from round, may or may not be over
        assert game.phase in [GamePhase.GAME_OVER, GamePhase.ROUND_END]

    def test_game_over_explicit(self):
        """Cover explicit game over transition (lines 590-591)."""
        game = HeartsGame(["P0", "P1", "P2", "P3"])
        game.start_round()
        game.phase = GamePhase.PLAYING
        game._start_first_trick()

        # Set player 0's score at 99, so ANY points pushes over 100
        game.players[0].total_score = 99

        # Set up for the last trick of the round
        game.tricks_played = 12

        # Player 0 leads and will take this trick with points
        game.current_trick = Trick(lead_player_index=0, num_players=4)
        game.current_player_index = 0

        # Give player 0 a high club to win, others dump hearts
        game.players[0].hand = [Card(Suit.CLUBS, Rank.ACE)]
        game.players[1].hand = [Card(Suit.HEARTS, Rank.TWO)]
        game.players[2].hand = [Card(Suit.HEARTS, Rank.THREE)]
        game.players[3].hand = [Card(Suit.HEARTS, Rank.FOUR)]

        # Player 0 leads
        result = game.play_card(0, Card(Suit.CLUBS, Rank.ACE))
        assert result["valid"]

        # Others dump hearts (can't follow suit)
        result = game.play_card(1, Card(Suit.HEARTS, Rank.TWO))
        assert result["valid"]

        result = game.play_card(2, Card(Suit.HEARTS, Rank.THREE))
        assert result["valid"]

        result = game.play_card(3, Card(Suit.HEARTS, Rank.FOUR))
        assert result["valid"]
        assert result["trick_complete"]
        assert result["round_complete"]
        assert result["game_over"]

        # Must be GAME_OVER since player 0 now has 99 + 3 = 102 points
        assert game.phase == GamePhase.GAME_OVER


class TestMoonShootingDiscardNoLowHearts:
    """Test moon shooting discard when no low hearts available (line 903)."""

    def test_moon_shooting_discard_no_low_hearts(self):
        """Cover return max safe_discards when shooting moon with no low hearts."""
        weights = AIWeights(
            control_card_count=2,
            moon_attempt_threshold=30,
            moon_high_hearts_weight=20,
            moon_queen_spades_weight=15,
        )
        ai = HeartsAI(weights=weights)
        ai.reset_round(4)

        # Strong hand for moon attempt with heart control but NO low hearts to dump
        hand = [
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.HEARTS, Rank.KING),
            Card(Suit.HEARTS, Rank.QUEEN),  # High hearts only
            Card(Suit.SPADES, Rank.QUEEN),
            Card(Suit.CLUBS, Rank.KING),  # Safe discard option
            Card(Suit.CLUBS, Rank.THREE),  # Safe discard option
        ]

        trick = Trick(lead_player_index=0, num_players=4)
        trick.add_card(0, Card(Suit.DIAMONDS, Rank.ACE))

        # Only non-heart, non-QoS cards are valid plays (safe discards)
        valid_plays = [
            Card(Suit.CLUBS, Rank.KING),
            Card(Suit.CLUBS, Rank.THREE),
        ]

        play = ai.select_play(hand, valid_plays, trick, 1, 4, True)

        # Should discard highest safe card since no low hearts in valid_plays
        assert play == Card(Suit.CLUBS, Rank.KING)


# =============================================================================
# JSON Weights Loading Tests (hearts_ai.py lines 247, 262-265)
# =============================================================================


class TestLoadWeightsFromJson:
    """Tests for load_weights exception handling."""

    def test_load_weights_file_not_exists(self, tmp_path):
        """Cover file not exists path (line 247)."""
        from hearts_ai import load_weights

        nonexistent = tmp_path / "nonexistent_weights.json"
        result = load_weights(str(nonexistent))

        # Should return defaults when file doesn't exist
        assert GameMode.PLAYER_4 in result
        assert isinstance(result[GameMode.PLAYER_4], AIWeights)

    def test_load_weights_invalid_json(self, tmp_path):
        """Cover invalid JSON file (line 262)."""
        from hearts_ai import load_weights

        invalid_json = tmp_path / "invalid.json"
        invalid_json.write_text("{ invalid json }")

        # Should handle gracefully and return defaults
        result = load_weights(str(invalid_json))
        assert GameMode.PLAYER_4 in result

    def test_load_weights_invalid_gamemode(self, tmp_path):
        """Cover invalid GameMode key (line 263)."""
        from hearts_ai import load_weights
        import json

        weights_file = tmp_path / "weights.json"
        data = {
            "INVALID_MODE": {"pass_high_cards": 75.0},
            "PLAYER_4": {"pass_queen_of_spades": 110.0},
        }
        weights_file.write_text(json.dumps(data))

        result = load_weights(str(weights_file))
        # Should skip invalid mode and load PLAYER_4
        assert result[GameMode.PLAYER_4].pass_queen_of_spades == 110.0

    def test_load_weights_invalid_weight_field(self, tmp_path):
        """Cover invalid weight field name (line 264)."""
        from hearts_ai import load_weights
        import json

        weights_file = tmp_path / "weights.json"
        data = {
            "PLAYER_4": {
                "pass_queen_of_spades": 110.0,
                "nonexistent_field": 999.0,  # Invalid field
                "pass_high_cards": 75.0,
            }
        }
        weights_file.write_text(json.dumps(data))

        result = load_weights(str(weights_file))
        # Should load valid fields and skip invalid ones
        assert result[GameMode.PLAYER_4].pass_queen_of_spades == 110.0
        assert result[GameMode.PLAYER_4].pass_high_cards == 75.0

    def test_load_weights_invalid_weight_value(self, tmp_path):
        """Cover invalid weight value type (line 264-265)."""
        from hearts_ai import load_weights
        import json

        weights_file = tmp_path / "weights.json"
        data = {
            "PLAYER_4": {
                "pass_queen_of_spades": "invalid_string",  # Wrong type
            }
        }
        weights_file.write_text(json.dumps(data))

        # Should handle ValueError from AIWeights constructor
        result = load_weights(str(weights_file))
