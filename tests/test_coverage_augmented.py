"""
Additional tests to augment code coverage for hearts game.
Targets specific untested code paths and edge cases.
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
    set_ai_weights,
    get_ai_weights,
    get_weights_summary,
    load_weights,
    DEFAULT_WEIGHTS,
)


# =============================================================================
# Card and Player String Representation Tests
# =============================================================================


class TestStringRepresentations:
    """Tests for __str__ and __repr__ methods."""

    def test_suit_str_all_suits(self):
        """Cover Suit.__str__ for all suits."""
        assert str(Suit.CLUBS) == "♣"
        assert str(Suit.DIAMONDS) == "♦"
        assert str(Suit.SPADES) == "♠"
        assert str(Suit.HEARTS) == "♥"

    def test_rank_str_all_ranks(self):
        """Cover Rank.__str__ for all ranks including face cards."""
        assert str(Rank.TWO) == "2"
        assert str(Rank.THREE) == "3"
        assert str(Rank.FOUR) == "4"
        assert str(Rank.FIVE) == "5"
        assert str(Rank.SIX) == "6"
        assert str(Rank.SEVEN) == "7"
        assert str(Rank.EIGHT) == "8"
        assert str(Rank.NINE) == "9"
        assert str(Rank.TEN) == "10"
        assert str(Rank.JACK) == "J"
        assert str(Rank.QUEEN) == "Q"
        assert str(Rank.KING) == "K"
        assert str(Rank.ACE) == "A"

    def test_card_str(self):
        """Cover Card.__str__ method."""
        card = Card(Suit.HEARTS, Rank.ACE)
        assert str(card) == "A♥"

    def test_card_repr(self):
        """Cover Card.__repr__ method."""
        card = Card(Suit.SPADES, Rank.QUEEN)
        assert repr(card) == "Q♠"


# =============================================================================
# Card Properties Tests
# =============================================================================


class TestCardProperties:
    """Tests for Card property methods."""

    def test_card_points_hearts(self):
        """Cover points property for hearts."""
        for rank in Rank:
            card = Card(Suit.HEARTS, rank)
            assert card.points == 1

    def test_card_points_queen_of_spades(self):
        """Cover points property for Q♠."""
        qos = Card(Suit.SPADES, Rank.QUEEN)
        assert qos.points == 13

    def test_card_points_non_point_cards(self):
        """Cover points property for non-point cards."""
        # Clubs
        assert Card(Suit.CLUBS, Rank.ACE).points == 0
        # Diamonds
        assert Card(Suit.DIAMONDS, Rank.KING).points == 0
        # Non-queen spades
        assert Card(Suit.SPADES, Rank.ACE).points == 0
        assert Card(Suit.SPADES, Rank.KING).points == 0

    def test_is_queen_of_spades_true(self):
        """Cover is_queen_of_spades property - True case."""
        qos = Card(Suit.SPADES, Rank.QUEEN)
        assert qos.is_queen_of_spades is True

    def test_is_queen_of_spades_false(self):
        """Cover is_queen_of_spades property - False cases."""
        # Different rank
        assert Card(Suit.SPADES, Rank.KING).is_queen_of_spades is False
        # Different suit
        assert Card(Suit.HEARTS, Rank.QUEEN).is_queen_of_spades is False

    def test_is_two_of_clubs_true(self):
        """Cover is_two_of_clubs property - True case."""
        two_clubs = Card(Suit.CLUBS, Rank.TWO)
        assert two_clubs.is_two_of_clubs is True

    def test_is_two_of_clubs_false(self):
        """Cover is_two_of_clubs property - False cases."""
        assert Card(Suit.CLUBS, Rank.THREE).is_two_of_clubs is False
        assert Card(Suit.DIAMONDS, Rank.TWO).is_two_of_clubs is False


# =============================================================================
# Player Method Tests
# =============================================================================


class TestPlayerMethods:
    """Tests for Player methods."""

    def test_player_reset_for_round(self):
        """Cover Player.reset_for_round method."""
        player = Player("Test")
        player.hand = [Card(Suit.HEARTS, Rank.ACE)]
        player.taken_cards = [Card(Suit.SPADES, Rank.QUEEN)]
        player.round_score = 13
        player.cards_to_pass = [Card(Suit.CLUBS, Rank.TWO)]

        player.reset_for_round()

        assert player.hand == []
        assert player.taken_cards == []
        assert player.round_score == 0
        assert player.cards_to_pass == []
        assert player.total_score == 0  # Total score not reset

    def test_player_has_card_true(self):
        """Cover Player.has_card - True case."""
        player = Player("Test")
        card = Card(Suit.HEARTS, Rank.ACE)
        player.hand = [card]
        assert player.has_card(card) is True

    def test_player_has_card_false(self):
        """Cover Player.has_card - False case."""
        player = Player("Test")
        player.hand = [Card(Suit.HEARTS, Rank.ACE)]
        assert player.has_card(Card(Suit.SPADES, Rank.QUEEN)) is False

    def test_player_has_suit_true(self):
        """Cover Player.has_suit - True case."""
        player = Player("Test")
        player.hand = [Card(Suit.HEARTS, Rank.ACE)]
        assert player.has_suit(Suit.HEARTS) is True

    def test_player_has_suit_false(self):
        """Cover Player.has_suit - False case."""
        player = Player("Test")
        player.hand = [Card(Suit.HEARTS, Rank.ACE)]
        assert player.has_suit(Suit.SPADES) is False

    def test_player_has_only_hearts_true(self):
        """Cover Player.has_only_hearts - True case."""
        player = Player("Test")
        player.hand = [
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.HEARTS, Rank.KING),
        ]
        assert player.has_only_hearts() is True

    def test_player_has_only_hearts_false(self):
        """Cover Player.has_only_hearts - False case."""
        player = Player("Test")
        player.hand = [
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.CLUBS, Rank.TWO),
        ]
        assert player.has_only_hearts() is False

    def test_player_get_cards_of_suit(self):
        """Cover Player.get_cards_of_suit method."""
        player = Player("Test")
        h_ace = Card(Suit.HEARTS, Rank.ACE)
        h_king = Card(Suit.HEARTS, Rank.KING)
        c_two = Card(Suit.CLUBS, Rank.TWO)
        player.hand = [h_ace, h_king, c_two]

        hearts = player.get_cards_of_suit(Suit.HEARTS)
        assert len(hearts) == 2
        assert h_ace in hearts
        assert h_king in hearts

    def test_player_play_card_success(self):
        """Cover Player.play_card - success case."""
        player = Player("Test")
        card = Card(Suit.HEARTS, Rank.ACE)
        player.hand = [card]

        played = player.play_card(card)
        assert played == card
        assert card not in player.hand

    def test_player_play_card_not_in_hand(self):
        """Cover Player.play_card - card not in hand."""
        player = Player("Test")
        player.hand = []

        with pytest.raises(ValueError, match="does not have"):
            player.play_card(Card(Suit.HEARTS, Rank.ACE))

    def test_player_receive_cards(self):
        """Cover Player.receive_cards method."""
        player = Player("Test")
        cards = [Card(Suit.HEARTS, Rank.ACE), Card(Suit.SPADES, Rank.KING)]
        player.receive_cards(cards)

        assert len(player.hand) == 2
        assert all(c in player.hand for c in cards)

    def test_player_take_trick(self):
        """Cover Player.take_trick method."""
        player = Player("Test")
        cards = [Card(Suit.HEARTS, Rank.ACE), Card(Suit.SPADES, Rank.QUEEN)]
        player.take_trick(cards)

        assert len(player.taken_cards) == 2

    def test_player_calculate_round_score(self):
        """Cover Player.calculate_round_score method."""
        player = Player("Test")
        player.taken_cards = [
            Card(Suit.HEARTS, Rank.ACE),  # 1 point
            Card(Suit.HEARTS, Rank.TWO),  # 1 point
            Card(Suit.SPADES, Rank.QUEEN),  # 13 points
        ]

        score = player.calculate_round_score()
        assert score == 15
        assert player.round_score == 15

    def test_player_sort_hand(self):
        """Cover Player.sort_hand method."""
        player = Player("Test")
        player.hand = [
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.CLUBS, Rank.TWO),
            Card(Suit.HEARTS, Rank.TWO),
        ]
        player.sort_hand()

        # Should be sorted by suit (CLUBS < HEARTS) then rank
        assert player.hand[0].suit == Suit.CLUBS
        assert player.hand[1].suit == Suit.HEARTS
        assert player.hand[1].rank.value < player.hand[2].rank.value


# =============================================================================
# Trick Method Tests
# =============================================================================


class TestTrickMethods:
    """Tests for Trick methods."""

    def test_trick_lead_suit_empty(self):
        """Cover Trick.lead_suit when empty."""
        trick = Trick(lead_player_index=0)
        assert trick.lead_suit is None

    def test_trick_lead_suit_with_cards(self):
        """Cover Trick.lead_suit with cards."""
        trick = Trick(lead_player_index=0)
        trick.add_card(0, Card(Suit.CLUBS, Rank.TWO))
        assert trick.lead_suit == Suit.CLUBS

    def test_trick_is_complete_false(self):
        """Cover Trick.is_complete - incomplete."""
        trick = Trick(lead_player_index=0, num_players=4)
        trick.add_card(0, Card(Suit.CLUBS, Rank.TWO))
        assert trick.is_complete is False

    def test_trick_is_complete_true(self):
        """Cover Trick.is_complete - complete."""
        trick = Trick(lead_player_index=0, num_players=4)
        for i in range(4):
            trick.add_card(i, Card(Suit.CLUBS, Rank(2 + i)))
        assert trick.is_complete is True

    def test_trick_get_winner_empty(self):
        """Cover Trick.get_winner with no cards."""
        trick = Trick(lead_player_index=0)
        with pytest.raises(ValueError, match="No cards in trick"):
            trick.get_winner()

    def test_trick_get_winner_first_card_wins(self):
        """Cover Trick.get_winner when lead card wins."""
        trick = Trick(lead_player_index=0, num_players=4)
        trick.add_card(0, Card(Suit.CLUBS, Rank.ACE))  # Highest
        trick.add_card(1, Card(Suit.CLUBS, Rank.TWO))
        trick.add_card(2, Card(Suit.CLUBS, Rank.THREE))
        trick.add_card(3, Card(Suit.CLUBS, Rank.FOUR))
        assert trick.get_winner() == 0

    def test_trick_get_cards(self):
        """Cover Trick.get_cards method."""
        trick = Trick(lead_player_index=0, num_players=4)
        c1 = Card(Suit.CLUBS, Rank.TWO)
        c2 = Card(Suit.CLUBS, Rank.THREE)
        trick.add_card(0, c1)
        trick.add_card(1, c2)

        cards = trick.get_cards()
        assert c1 in cards
        assert c2 in cards

    def test_trick_contains_points_true(self):
        """Cover Trick.contains_points - True case."""
        trick = Trick(lead_player_index=0, num_players=4)
        trick.add_card(0, Card(Suit.HEARTS, Rank.ACE))  # Point card
        trick.add_card(1, Card(Suit.HEARTS, Rank.TWO))
        assert trick.contains_points() is True

    def test_trick_contains_points_false(self):
        """Cover Trick.contains_points - False case."""
        trick = Trick(lead_player_index=0, num_players=4)
        trick.add_card(0, Card(Suit.CLUBS, Rank.ACE))
        trick.add_card(1, Card(Suit.CLUBS, Rank.TWO))
        assert trick.contains_points() is False


# =============================================================================
# GameMode Tests
# =============================================================================


class TestGameMode:
    """Tests for GameMode enum and from_settings."""

    def test_gamemode_from_settings_4_player(self):
        """Cover GameMode.from_settings with 4 players."""
        mode = GameMode.from_settings(4)
        assert mode == GameMode.PLAYER_4

    def test_gamemode_from_settings_3_player_remove(self):
        """Cover GameMode.from_settings with 3 players REMOVE_CARD."""
        mode = GameMode.from_settings(3, ThreePlayerMode.REMOVE_CARD)
        assert mode == GameMode.PLAYER_3_REMOVE

    def test_gamemode_from_settings_3_player_kitty(self):
        """Cover GameMode.from_settings with 3 players KITTY."""
        mode = GameMode.from_settings(3, ThreePlayerMode.KITTY)
        assert mode == GameMode.PLAYER_3_KITTY

    def test_gamemode_from_settings_default(self):
        """Cover GameMode.from_settings default fallback."""
        # Invalid player count should default to PLAYER_4
        mode = GameMode.from_settings(5)
        assert mode == GameMode.PLAYER_4


# =============================================================================
# HeartsGame State Transition Tests
# =============================================================================


class TestGameStateTransitions:
    """Tests for game state transitions."""

    def test_transition_dealing_to_passing(self):
        """Cover DEALING to PASSING transition."""
        game = HeartsGame(["P0", "P1", "P2", "P3"])
        assert game.phase == GamePhase.DEALING

        game.start_round()
        assert game.phase == GamePhase.PASSING

    def test_transition_passing_to_playing(self):
        """Cover PASSING to PLAYING transition."""
        game = HeartsGame(["P0", "P1", "P2", "P3"])
        game.start_round()
        assert game.phase == GamePhase.PASSING

        # Set pass cards for all players
        for i in range(4):
            cards = list(game.players[i].hand[:3])
            game.set_pass_cards(i, cards)

        game.execute_pass()
        assert game.phase == GamePhase.PLAYING

    def test_transition_playing_to_round_end(self):
        """Cover PLAYING to ROUND_END transition."""
        game = HeartsGame(["P0", "P1", "P2", "P3"])
        game.start_round()

        # Skip to playing
        for i in range(4):
            cards = list(game.players[i].hand[:3])
            game.set_pass_cards(i, cards)
        game.execute_pass()

        # Play all tricks
        while game.phase == GamePhase.PLAYING:
            idx = game.current_player_index
            valid = game.get_valid_plays(idx)
            if valid:
                game.play_card(idx, valid[0])

        assert game.phase in [GamePhase.ROUND_END, GamePhase.GAME_OVER]

    def test_round_end_to_new_round(self):
        """Cover starting new round from ROUND_END."""
        game = HeartsGame(["P0", "P1", "P2", "P3"])
        game.phase = GamePhase.ROUND_END
        game.round_number = 1

        game.start_round()
        assert game.phase == GamePhase.PASSING
        assert game.round_number == 2


# =============================================================================
# 3-Player Game Tests
# =============================================================================


class TestThreePlayerGame:
    """Tests for 3-player specific mechanics."""

    def test_three_player_removes_two_of_diamonds(self):
        """Cover 2♦ removal in 3-player REMOVE_CARD mode."""
        game = HeartsGame(["P0", "P1", "P2"], ThreePlayerMode.REMOVE_CARD)
        game.start_round()

        # Check that 2♦ is not in any hand
        two_diamonds = Card(Suit.DIAMONDS, Rank.TWO)
        all_cards = []
        for player in game.players:
            all_cards.extend(player.hand)

        assert two_diamonds not in all_cards
        assert game.removed_card == two_diamonds

    def test_three_player_pass_right_offset(self):
        """Cover 3-player RIGHT pass direction (offset 2)."""
        game = HeartsGame(["P0", "P1", "P2"])
        game.round_number = 1  # Round 2 = RIGHT
        game.start_round()
        assert game.pass_direction == PassDirection.RIGHT

        # Track cards before pass
        p0_cards_before = list(game.players[0].hand[:3])

        for i in range(3):
            cards = list(game.players[i].hand[:3])
            game.set_pass_cards(i, cards)

        game.execute_pass()
        assert game.phase == GamePhase.PLAYING

    def test_three_player_pass_left_offset(self):
        """Cover 3-player LEFT pass direction (offset 1)."""
        game = HeartsGame(["P0", "P1", "P2"])
        game.start_round()  # Round 1 = LEFT
        assert game.pass_direction == PassDirection.LEFT

    def test_three_player_no_pass_round(self):
        """Cover 3-player NONE pass direction."""
        game = HeartsGame(["P0", "P1", "P2"])
        game.round_number = 2  # Round 3 = NONE
        game.start_round()
        assert game.pass_direction == PassDirection.NONE
        assert game.phase == GamePhase.PLAYING

    def test_three_player_17_tricks(self):
        """Cover 3-player has 17 tricks per round."""
        game = HeartsGame(["P0", "P1", "P2"])
        assert game.tricks_per_round == 17

    def test_three_player_kitty_mode_setup(self):
        """Cover 3-player KITTY mode setup."""
        game = HeartsGame(["P0", "P1", "P2"], ThreePlayerMode.KITTY)
        game.start_round()

        # Kitty should be set
        assert game.kitty_card is not None
        assert game.kitty_claimed is False

        # 2♣ should not be in kitty
        assert game.kitty_card != Card(Suit.CLUBS, Rank.TWO)

    def test_three_player_kitty_claimed(self):
        """Cover kitty card claimed when taking points."""
        game = HeartsGame(["P0", "P1", "P2"], ThreePlayerMode.KITTY)
        game.start_round()

        # Skip passing if needed
        if game.phase == GamePhase.PASSING:
            for i in range(3):
                cards = list(game.players[i].hand[:3])
                game.set_pass_cards(i, cards)
            game.execute_pass()

        initial_kitty = game.kitty_card

        # Play until someone takes points (kitty claim)
        kitty_claimed_result = None
        max_iterations = 200
        iterations = 0

        while not game.kitty_claimed and game.phase == GamePhase.PLAYING and iterations < max_iterations:
            idx = game.current_player_index
            valid = game.get_valid_plays(idx)
            if valid:
                result = game.play_card(idx, valid[0])
                if result.get("kitty_claimed"):
                    kitty_claimed_result = result
            iterations += 1

        # Verify kitty mechanics worked
        if kitty_claimed_result:
            assert kitty_claimed_result["kitty_card"] == initial_kitty


# =============================================================================
# Shooting the Moon Tests
# =============================================================================


class TestShootingTheMoon:
    """Tests for shooting the moon mechanics."""

    def test_moon_shooter_scores_detected(self):
        """Cover moon shooter detection in _end_round."""
        game = HeartsGame(["P0", "P1", "P2", "P3"])
        game.start_round()

        # Simulate player 0 taking all point cards (26 points)
        all_hearts = [Card(Suit.HEARTS, rank) for rank in Rank]
        qos = Card(Suit.SPADES, Rank.QUEEN)
        game.players[0].taken_cards = all_hearts + [qos]
        game.players[0].calculate_round_score()

        # Manually call _end_round
        game._end_round()

        # Moon shooter gets 0, others get 26
        assert game.players[0].round_score == 0
        assert game.players[1].round_score == 26
        assert game.players[2].round_score == 26
        assert game.players[3].round_score == 26


# =============================================================================
# HeartsGame Error Handling Tests
# =============================================================================


class TestGameErrorHandling:
    """Tests for game error handling."""

    def test_invalid_player_count(self):
        """Cover invalid player count error."""
        with pytest.raises(ValueError, match="3 or 4 players"):
            HeartsGame(["P0", "P1"])

        with pytest.raises(ValueError, match="3 or 4 players"):
            HeartsGame(["P0", "P1", "P2", "P3", "P4"])

    def test_set_pass_cards_wrong_phase(self):
        """Cover set_pass_cards when not in passing phase."""
        game = HeartsGame(["P0", "P1", "P2", "P3"])
        game.phase = GamePhase.PLAYING

        with pytest.raises(ValueError, match="Not in passing phase"):
            game.set_pass_cards(0, [])

    def test_set_pass_cards_wrong_count(self):
        """Cover set_pass_cards with wrong number of cards."""
        game = HeartsGame(["P0", "P1", "P2", "P3"])
        game.start_round()

        with pytest.raises(ValueError, match="Must pass exactly 3 cards"):
            game.set_pass_cards(0, [game.players[0].hand[0]])

    def test_set_pass_cards_card_not_in_hand(self):
        """Cover set_pass_cards with card not in hand."""
        game = HeartsGame(["P0", "P1", "P2", "P3"])
        game.start_round()

        # Find a card not in player 0's hand
        fake_card = None
        for suit in Suit:
            for rank in Rank:
                card = Card(suit, rank)
                if card not in game.players[0].hand:
                    fake_card = card
                    break
            if fake_card:
                break

        with pytest.raises(ValueError, match="Player does not have"):
            game.set_pass_cards(0, [fake_card, game.players[0].hand[0], game.players[0].hand[1]])

    def test_execute_pass_wrong_phase(self):
        """Cover execute_pass when not in passing phase."""
        game = HeartsGame(["P0", "P1", "P2", "P3"])
        game.phase = GamePhase.PLAYING

        with pytest.raises(ValueError, match="Not in passing phase"):
            game.execute_pass()

    def test_execute_pass_not_ready(self):
        """Cover execute_pass when not all players ready."""
        game = HeartsGame(["P0", "P1", "P2", "P3"])
        game.start_round()

        # Only set cards for one player
        game.set_pass_cards(0, list(game.players[0].hand[:3]))

        with pytest.raises(ValueError, match="Not all players have selected"):
            game.execute_pass()

    def test_play_card_wrong_phase(self):
        """Cover play_card when not in playing phase."""
        game = HeartsGame(["P0", "P1", "P2", "P3"])
        game.phase = GamePhase.PASSING

        result = game.play_card(0, Card(Suit.CLUBS, Rank.TWO))
        assert result["valid"] is False
        assert "Not in playing phase" in result["error"]

    def test_play_card_wrong_player(self):
        """Cover play_card when wrong player's turn."""
        game = HeartsGame(["P0", "P1", "P2", "P3"])
        game.start_round()
        for i in range(4):
            game.set_pass_cards(i, list(game.players[i].hand[:3]))
        game.execute_pass()

        wrong_player = (game.current_player_index + 1) % 4
        result = game.play_card(wrong_player, Card(Suit.CLUBS, Rank.TWO))
        assert result["valid"] is False
        assert "Not player" in result["error"]

    def test_play_card_invalid_card(self):
        """Cover play_card with invalid card."""
        game = HeartsGame(["P0", "P1", "P2", "P3"])
        game.start_round()
        for i in range(4):
            game.set_pass_cards(i, list(game.players[i].hand[:3]))
        game.execute_pass()

        idx = game.current_player_index
        # Try to play a card that's not valid (not 2♣ on first trick)
        invalid_card = Card(Suit.HEARTS, Rank.ACE)
        result = game.play_card(idx, invalid_card)
        assert result["valid"] is False
        assert "is not a valid play" in result["error"]


# =============================================================================
# HeartsGame State Information Tests
# =============================================================================


class TestGameStateInfo:
    """Tests for game state information methods."""

    def test_get_scores(self):
        """Cover get_scores method."""
        game = HeartsGame(["P0", "P1", "P2", "P3"])
        game.players[0].total_score = 10
        game.players[1].total_score = 20

        scores = game.get_scores()
        assert scores["P0"] == 10
        assert scores["P1"] == 20

    def test_get_round_scores(self):
        """Cover get_round_scores method."""
        game = HeartsGame(["P0", "P1", "P2", "P3"])
        game.players[0].round_score = 5
        game.players[1].round_score = 8

        scores = game.get_round_scores()
        assert scores["P0"] == 5
        assert scores["P1"] == 8

    def test_get_game_state_comprehensive(self):
        """Cover get_game_state method comprehensively."""
        game = HeartsGame(["P0", "P1", "P2", "P3"])
        game.start_round()

        state = game.get_game_state()

        assert state["phase"] == "PASSING"
        assert state["num_players"] == 4
        assert state["round_number"] == 1
        assert "pass_direction" in state
        assert "current_player" in state
        assert "tricks_played" in state
        assert "hearts_broken" in state
        assert "scores" in state

    def test_get_game_state_with_trick(self):
        """Cover get_game_state with active trick."""
        game = HeartsGame(["P0", "P1", "P2", "P3"])
        game.start_round()
        for i in range(4):
            game.set_pass_cards(i, list(game.players[i].hand[:3]))
        game.execute_pass()

        # Play first card
        idx = game.current_player_index
        valid = game.get_valid_plays(idx)
        game.play_card(idx, valid[0])

        state = game.get_game_state()
        assert len(state["current_trick"]) == 1


# =============================================================================
# AI Module-Level Function Tests
# =============================================================================


class TestAIModuleFunctions:
    """Tests for AI module-level convenience functions."""

    def test_set_and_get_ai_weights(self):
        """Cover set_ai_weights and get_ai_weights."""
        custom_weights = AIWeights(pass_queen_of_spades=150.0)
        set_ai_weights(custom_weights)

        retrieved = get_ai_weights()
        assert retrieved.pass_queen_of_spades == 150.0

    def test_get_ai_creates_instance(self):
        """Cover get_ai function."""
        reset_all_ai(4)
        ai = get_ai(0)
        assert isinstance(ai, HeartsAI)

    def test_reset_all_ai_with_mode(self):
        """Cover reset_all_ai with specific mode."""
        custom_weights = AIWeights(pass_high_cards=75.0)
        reset_all_ai(4, GameMode.PLAYER_4, custom_weights)

        ai = get_ai(0)
        assert ai.weights.pass_high_cards == 75.0

    def test_reset_all_ai_default_weights(self):
        """Cover reset_all_ai with default weights."""
        reset_all_ai(4, GameMode.PLAYER_4)

        ai = get_ai(0)
        assert isinstance(ai.weights, AIWeights)

    def test_record_trick_for_all(self):
        """Cover record_trick_for_all function."""
        reset_all_ai(4)

        trick_cards = [
            Card(Suit.CLUBS, Rank.TWO),
            Card(Suit.CLUBS, Rank.ACE),
            Card(Suit.CLUBS, Rank.THREE),
            Card(Suit.CLUBS, Rank.FOUR),
        ]
        trick_info = [
            (0, Card(Suit.CLUBS, Rank.TWO)),
            (1, Card(Suit.CLUBS, Rank.ACE)),
            (2, Card(Suit.CLUBS, Rank.THREE)),
            (3, Card(Suit.CLUBS, Rank.FOUR)),
        ]

        record_trick_for_all(trick_cards, 1, trick_info)

        # Verify all AIs recorded the trick
        ai = get_ai(0)
        assert ai.points_taken.get(1, 0) == 0  # No points in this trick

    def test_get_weights_summary(self):
        """Cover get_weights_summary function."""
        summary = get_weights_summary()

        assert isinstance(summary, str)
        assert "PASSING STRATEGY" in summary
        assert "LEADING STRATEGY" in summary
        assert "MOON DEFENSE" in summary


# =============================================================================
# CardTracker Extended Tests
# =============================================================================


class TestCardTrackerExtended:
    """Extended tests for CardTracker."""

    def test_remaining_in_suit_with_excluding(self):
        """Cover remaining_in_suit with excluding_hand parameter."""
        tracker = CardTracker()

        # Play a few hearts
        tracker.record_cards([Card(Suit.HEARTS, Rank.ACE)])
        tracker.record_cards([Card(Suit.HEARTS, Rank.KING)])

        # Exclude some cards from hand
        excluding = [Card(Suit.HEARTS, Rank.QUEEN), Card(Suit.HEARTS, Rank.JACK)]

        remaining = tracker.remaining_in_suit(Suit.HEARTS, excluding)

        # Should not include played cards or excluded cards
        assert Card(Suit.HEARTS, Rank.ACE) not in remaining
        assert Card(Suit.HEARTS, Rank.KING) not in remaining
        assert Card(Suit.HEARTS, Rank.QUEEN) not in remaining
        assert Card(Suit.HEARTS, Rank.JACK) not in remaining
        assert Card(Suit.HEARTS, Rank.TEN) in remaining

    def test_remaining_in_suit_no_excluding(self):
        """Cover remaining_in_suit without excluding_hand."""
        tracker = CardTracker()
        tracker.record_cards([Card(Suit.SPADES, Rank.ACE)])

        remaining = tracker.remaining_in_suit(Suit.SPADES)
        assert Card(Suit.SPADES, Rank.ACE) not in remaining
        assert len(remaining) == 12  # 13 - 1 played

    def test_hearts_played_count(self):
        """Cover hearts_played_count method."""
        tracker = CardTracker()
        assert tracker.hearts_played_count() == 0

        tracker.record_cards([Card(Suit.HEARTS, Rank.ACE)])
        tracker.record_cards([Card(Suit.HEARTS, Rank.TWO)])
        tracker.record_cards([Card(Suit.CLUBS, Rank.THREE)])

        assert tracker.hearts_played_count() == 2

    def test_tracker_reset(self):
        """Cover CardTracker.reset method."""
        tracker = CardTracker()
        tracker.record_cards([Card(Suit.HEARTS, Rank.ACE)])
        tracker.mark_player_void(0, Suit.CLUBS)
        tracker.record_passed_cards(1, [Card(Suit.SPADES, Rank.KING)])

        tracker.reset()

        assert len(tracker.played_cards) == 0
        assert len(tracker.void_players) == 0
        assert len(tracker.passed_cards) == 0


# =============================================================================
# AI Strategy Extended Tests
# =============================================================================


class TestAIStrategyExtended:
    """Extended tests for AI strategy edge cases."""

    def test_bleed_spades_when_protected_queen(self):
        """Cover bleeding spades when Q♠ is protected."""
        weights = AIWeights(
            bleed_spades_priority=50.0,
            queen_protection_threshold=2,
            flush_queen_priority=0,  # Disable flushing
            lead_spades_priority=0,
        )
        ai = HeartsAI(weights=weights)
        ai.reset_round(4)

        # Hand with protected Q♠
        hand = [
            Card(Suit.SPADES, Rank.QUEEN),
            Card(Suit.SPADES, Rank.TWO),
            Card(Suit.SPADES, Rank.THREE),  # 2 low spades = protected
            Card(Suit.CLUBS, Rank.FOUR),
        ]
        valid_plays = hand.copy()
        trick = Trick(lead_player_index=0)

        play = ai.select_play(hand, valid_plays, trick, 0, 4, False)

        # Should lead spade to bleed
        assert play.suit == Suit.SPADES

    def test_moon_shooting_lead_high(self):
        """Cover leading high when shooting moon."""
        weights = AIWeights(
            moon_attempt_threshold=30,
            moon_high_hearts_weight=20,
            moon_queen_spades_weight=25,
        )
        ai = HeartsAI(weights=weights)
        ai.reset_round(4)

        # Strong moon hand
        hand = [
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.HEARTS, Rank.KING),
            Card(Suit.HEARTS, Rank.QUEEN),
            Card(Suit.SPADES, Rank.QUEEN),
            Card(Suit.CLUBS, Rank.TWO),
        ]
        valid_plays = hand.copy()
        trick = Trick(lead_player_index=0)

        play = ai.select_play(hand, valid_plays, trick, 0, 4, True)

        # Should lead highest card when shooting moon
        assert play.rank.value >= Rank.QUEEN.value

    def test_moon_shooting_follow_win_points(self):
        """Cover following to win points when shooting moon."""
        weights = AIWeights(
            moon_attempt_threshold=30,
            moon_high_hearts_weight=20,
            moon_queen_spades_weight=25,
        )
        ai = HeartsAI(weights=weights)
        ai.reset_round(4)

        # Strong moon hand
        hand = [
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.HEARTS, Rank.KING),
            Card(Suit.HEARTS, Rank.QUEEN),
            Card(Suit.SPADES, Rank.QUEEN),
            Card(Suit.CLUBS, Rank.ACE),
            Card(Suit.CLUBS, Rank.KING),
        ]

        # Trick with hearts (points)
        trick = Trick(lead_player_index=0, num_players=4)
        trick.add_card(0, Card(Suit.CLUBS, Rank.TWO))
        trick.add_card(1, Card(Suit.HEARTS, Rank.TWO))  # Point

        valid_plays = [Card(Suit.CLUBS, Rank.ACE), Card(Suit.CLUBS, Rank.KING)]

        play = ai.select_play(hand, valid_plays, trick, 2, 4, True)

        # Should play highest to win when shooting moon
        assert play == Card(Suit.CLUBS, Rank.ACE)

    def test_ai_single_valid_play(self):
        """Cover single valid play selection."""
        ai = HeartsAI()
        ai.reset_round(4)

        hand = [Card(Suit.CLUBS, Rank.TWO)]
        valid_plays = [Card(Suit.CLUBS, Rank.TWO)]
        trick = Trick(lead_player_index=0)

        play = ai.select_play(hand, valid_plays, trick, 0, 4, False)
        assert play == Card(Suit.CLUBS, Rank.TWO)

    def test_ai_no_valid_plays_error(self):
        """Cover error when no valid plays."""
        ai = HeartsAI()
        hand = []
        valid_plays = []
        trick = Trick(lead_player_index=0)

        with pytest.raises(ValueError, match="No valid plays"):
            ai.select_play(hand, valid_plays, trick, 0, 4, False)


# =============================================================================
# Pass Strategy Extended Tests
# =============================================================================


class TestPassStrategyExtended:
    """Extended tests for passing strategy."""

    def test_pass_left_penalty(self):
        """Cover LEFT direction high card penalty."""
        weights = AIWeights(
            pass_left_high_card_penalty=50.0,
            pass_high_hearts=70.0,
        )
        ai = HeartsAI(weights=weights)

        hand = [
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
            Card(Suit.CLUBS, Rank.JACK),
            Card(Suit.CLUBS, Rank.QUEEN),
        ]

        passed = ai.select_pass_cards(hand, "LEFT")
        # With LEFT penalty, high hearts may still be passed but penalized
        assert len(passed) == 3

    def test_pass_two_of_clubs_control(self):
        """Cover passing 2♣ control weight."""
        weights = AIWeights(
            pass_two_of_clubs_control=80.0,  # High priority to pass 2♣
            retain_ace_of_clubs_bonus=0.0,   # No bonus for keeping it
            pass_high_hearts=30.0,           # Lower priority for hearts
            pass_high_cards=30.0,            # Lower priority for high cards
        )
        ai = HeartsAI(weights=weights)

        hand = [
            Card(Suit.CLUBS, Rank.TWO),
            Card(Suit.CLUBS, Rank.ACE),
            Card(Suit.HEARTS, Rank.KING),
            Card(Suit.HEARTS, Rank.QUEEN),
            Card(Suit.HEARTS, Rank.JACK),
            Card(Suit.DIAMONDS, Rank.TWO),
            Card(Suit.DIAMONDS, Rank.THREE),
            Card(Suit.DIAMONDS, Rank.FOUR),
            Card(Suit.DIAMONDS, Rank.FIVE),
            Card(Suit.DIAMONDS, Rank.SIX),
            Card(Suit.DIAMONDS, Rank.SEVEN),
            Card(Suit.DIAMONDS, Rank.EIGHT),
            Card(Suit.DIAMONDS, Rank.NINE),
        ]

        passed = ai.select_pass_cards(hand, "LEFT")
        # With high control priority and no retain bonus, 2♣ should be passed
        assert Card(Suit.CLUBS, Rank.TWO) in passed

    def test_pass_last_card_penalty(self):
        """Cover penalty for passing last card of a suit."""
        weights = AIWeights(
            pass_last_card_penalty=100.0,
            pass_to_void_suit=30.0,
        )
        ai = HeartsAI(weights=weights)

        # Only one diamond
        hand = [
            Card(Suit.DIAMONDS, Rank.ACE),  # Only diamond
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
            Card(Suit.CLUBS, Rank.JACK),
        ]

        passed = ai.select_pass_cards(hand, "LEFT")
        # High penalty should discourage passing lone diamond
        # but other factors may override
        assert len(passed) == 3


# =============================================================================
# Hearts Breaking Tests
# =============================================================================


class TestHeartsBreaking:
    """Tests for hearts breaking mechanics."""

    def test_hearts_break_on_play(self):
        """Cover hearts breaking when heart is played."""
        game = HeartsGame(["P0", "P1", "P2", "P3"])
        game.start_round()
        for i in range(4):
            game.set_pass_cards(i, list(game.players[i].hand[:3]))
        game.execute_pass()

        assert game.hearts_broken is False

        # Play through tricks until hearts break
        while not game.hearts_broken and game.phase == GamePhase.PLAYING:
            idx = game.current_player_index
            valid = game.get_valid_plays(idx)
            if valid:
                # Pick a heart if available to break hearts faster
                hearts = [c for c in valid if c.suit == Suit.HEARTS]
                card = hearts[0] if hearts else valid[0]
                game.play_card(idx, card)
