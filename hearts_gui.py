"""
Pygame interface for Hearts card game.

Run with: python hearts_gui.py
"""

import pygame
import sys
from typing import Optional
from hearts_game import HeartsGame, Card, Suit, Rank, GamePhase, ThreePlayerMode, Player
from hearts_ai import HeartsAI, reset_all_ai, record_trick_for_all, get_ai


# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (34, 139, 34)
DARK_GREEN = (0, 100, 0)
RED = (220, 20, 60)
BLUE = (65, 105, 225)
GOLD = (255, 215, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
CARD_WHITE = (252, 252, 248)
HIGHLIGHT = (255, 255, 150)
SELECTED = (150, 255, 150)

# Card dimensions
CARD_WIDTH = 71
CARD_HEIGHT = 96
CARD_SPACING = 25

# Window dimensions
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 768


class CardRenderer:
    """Renders playing cards using pygame drawing."""

    SUIT_SYMBOLS = {
        Suit.HEARTS: "\u2665",
        Suit.DIAMONDS: "\u2666",
        Suit.CLUBS: "\u2663",
        Suit.SPADES: "\u2660",
    }

    SUIT_COLORS = {
        Suit.HEARTS: RED,
        Suit.DIAMONDS: RED,
        Suit.CLUBS: BLACK,
        Suit.SPADES: BLACK,
    }

    RANK_SYMBOLS = {
        Rank.TWO: "2",
        Rank.THREE: "3",
        Rank.FOUR: "4",
        Rank.FIVE: "5",
        Rank.SIX: "6",
        Rank.SEVEN: "7",
        Rank.EIGHT: "8",
        Rank.NINE: "9",
        Rank.TEN: "10",
        Rank.JACK: "J",
        Rank.QUEEN: "Q",
        Rank.KING: "K",
        Rank.ACE: "A",
    }

    def __init__(self):
        pygame.font.init()
        self.rank_font = pygame.font.SysFont("segoeuisymbol", 20, bold=True)
        self.suit_font = pygame.font.SysFont("segoeuisymbol", 24)
        self.center_font = pygame.font.SysFont("segoeuisymbol", 36)
        self._cache: dict[tuple, pygame.Surface] = {}

    def render_card(
        self,
        card: Card,
        highlighted: bool = False,
        selected: bool = False,
        small: bool = False,
    ) -> pygame.Surface:
        """Render a card as a pygame surface."""
        cache_key = (card.suit, card.rank, highlighted, selected, small)
        if cache_key in self._cache:
            return self._cache[cache_key]

        width = CARD_WIDTH if not small else CARD_WIDTH // 2
        height = CARD_HEIGHT if not small else CARD_HEIGHT // 2

        surface = pygame.Surface((width, height), pygame.SRCALPHA)

        # Card background
        if selected:
            bg_color = SELECTED
        elif highlighted:
            bg_color = HIGHLIGHT
        else:
            bg_color = CARD_WHITE

        # Draw card shape with rounded corners
        pygame.draw.rect(surface, bg_color, (0, 0, width, height), border_radius=5)
        pygame.draw.rect(surface, BLACK, (0, 0, width, height), 2, border_radius=5)

        color = self.SUIT_COLORS[card.suit]
        rank_str = self.RANK_SYMBOLS[card.rank]
        suit_str = self.SUIT_SYMBOLS[card.suit]

        if small:
            font = pygame.font.SysFont("segoeuisymbol", 12, bold=True)
            # Just show rank and suit in corner
            text = font.render(f"{rank_str}{suit_str}", True, color)
            surface.blit(text, (3, 3))
        else:
            # Top-left rank and suit
            rank_text = self.rank_font.render(rank_str, True, color)
            suit_text = self.suit_font.render(suit_str, True, color)
            surface.blit(rank_text, (5, 2))
            surface.blit(suit_text, (5, 20))

            # Center suit
            center_text = self.center_font.render(suit_str, True, color)
            center_rect = center_text.get_rect(center=(width // 2, height // 2))
            surface.blit(center_text, center_rect)

            # Bottom-right rank and suit (rotated)
            rank_text_rot = pygame.transform.rotate(rank_text, 180)
            suit_text_rot = pygame.transform.rotate(suit_text, 180)
            surface.blit(
                rank_text_rot, (width - 20 - rank_text.get_width() // 2, height - 22)
            )
            surface.blit(
                suit_text_rot, (width - 22 - suit_text.get_width() // 2, height - 45)
            )

        self._cache[cache_key] = surface
        return surface

    def render_card_back(self, small: bool = False) -> pygame.Surface:
        """Render the back of a card."""
        width = CARD_WIDTH if not small else CARD_WIDTH // 2
        height = CARD_HEIGHT if not small else CARD_HEIGHT // 2

        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.rect(surface, BLUE, (0, 0, width, height), border_radius=5)
        pygame.draw.rect(surface, BLACK, (0, 0, width, height), 2, border_radius=5)

        # Draw pattern
        pygame.draw.rect(
            surface, WHITE, (4, 4, width - 8, height - 8), 1, border_radius=3
        )

        return surface


class Button:
    """Simple button class for UI."""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        text: str,
        font: pygame.font.Font,
        enabled: bool = True,
    ):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = font
        self.enabled = enabled
        self.hovered = False

    def draw(self, surface: pygame.Surface):
        if not self.enabled:
            color = GRAY
            text_color = LIGHT_GRAY
        elif self.hovered:
            color = GOLD
            text_color = BLACK
        else:
            color = WHITE
            text_color = BLACK

        pygame.draw.rect(surface, color, self.rect, border_radius=5)
        pygame.draw.rect(surface, BLACK, self.rect, 2, border_radius=5)

        text_surf = self.font.render(self.text, True, text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Returns True if button was clicked."""
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.enabled and self.rect.collidepoint(event.pos):
                return True
        return False


class HeartsGUI:
    """Main GUI class for Hearts game."""

    def __init__(
        self,
        num_players: int = 4,
        three_player_mode: ThreePlayerMode = ThreePlayerMode.REMOVE_CARD,
        auto_play: bool = False,
        play_delay: float = 1.0,
    ):
        pygame.init()
        title = "Hearts (Auto)" if auto_play else "Hearts"
        pygame.display.set_caption(title)

        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.card_renderer = CardRenderer()

        # Fonts
        self.title_font = pygame.font.SysFont("arial", 32, bold=True)
        self.label_font = pygame.font.SysFont("arial", 22, bold=True)
        self.score_font = pygame.font.SysFont("arial", 16)
        self.button_font = pygame.font.SysFont("arial", 18, bold=True)

        # Game state
        self.num_players = num_players
        self.three_player_mode = three_player_mode
        self.game: Optional[HeartsGame] = None
        self.human_player_index = 0

        # Auto-play mode
        self.auto_play = auto_play
        self.play_delay_ms = int(play_delay * 1000)  # Convert to milliseconds
        self.last_auto_action = 0  # Timestamp of last auto action

        # UI state
        self.selected_cards: list[Card] = []
        self.hovered_card: Optional[Card] = None
        self.card_rects: list[tuple[pygame.Rect, Card]] = []
        self.message = ""
        self.message_timer = 0

        # Animation state
        self.played_cards_display: list[tuple[int, Card]] = []
        self.trick_winner_display: Optional[int] = None
        self.display_timer = 0

        # Buttons
        self.pass_button = Button(
            WINDOW_WIDTH // 2 - 60,
            WINDOW_HEIGHT - 150,
            120,
            40,
            "Pass Cards",
            self.button_font,
        )
        self.new_game_button = Button(
            WINDOW_WIDTH // 2 - 80,
            WINDOW_HEIGHT // 2 + 150,
            160,
            50,
            "New Game",
            self.button_font,
        )
        self.continue_button = Button(
            WINDOW_WIDTH // 2 - 80,
            WINDOW_HEIGHT // 2 + 150,
            160,
            50,
            "Continue",
            self.button_font,
        )

        self.start_new_game()

    def start_new_game(self):
        """Start a new game."""
        if self.auto_play:
            if self.num_players == 4:
                names = ["South", "West", "North", "East"]
            else:
                names = ["South", "West", "East"]
        else:
            if self.num_players == 4:
                names = ["You", "West", "North", "East"]
            else:
                names = ["You", "Left", "Right"]

        self.game = HeartsGame(names, self.three_player_mode)
        self.game.start_round()
        self.selected_cards.clear()
        self.played_cards_display.clear()
        self.message = ""
        self.last_auto_action = pygame.time.get_ticks()

        # Reset AI for new game
        reset_all_ai(self.num_players, mode=self.game.game_mode)

    def set_message(self, msg: str, duration: int = 2000):
        """Set a temporary message to display."""
        self.message = msg
        self.message_timer = pygame.time.get_ticks() + duration

    def get_player_position(self, player_index: int) -> tuple[int, int, str]:
        """Get screen position and orientation for a player.

        Returns (x, y, orientation) where orientation is 'bottom', 'left', 'top', 'right'.
        """
        if self.num_players == 4:
            positions = {
                0: (WINDOW_WIDTH // 2, WINDOW_HEIGHT - 60, "bottom"),
                1: (80, WINDOW_HEIGHT // 2, "left"),
                2: (WINDOW_WIDTH // 2, 80, "top"),
                3: (WINDOW_WIDTH - 80, WINDOW_HEIGHT // 2, "right"),
            }
        else:
            positions = {
                0: (WINDOW_WIDTH // 2, WINDOW_HEIGHT - 60, "bottom"),
                1: (150, WINDOW_HEIGHT // 2 - 50, "left"),
                2: (WINDOW_WIDTH - 150, WINDOW_HEIGHT // 2 - 50, "right"),
            }
        return positions[player_index]

    def draw_background(self):
        """Draw the game table background."""
        self.screen.fill(DARK_GREEN)

        # Draw center play area
        center_rect = pygame.Rect(
            WINDOW_WIDTH // 2 - 150, WINDOW_HEIGHT // 2 - 100, 300, 200
        )
        pygame.draw.ellipse(self.screen, GREEN, center_rect)
        pygame.draw.ellipse(self.screen, DARK_GREEN, center_rect, 3)

    def draw_player_info(self):
        """Draw player names and scores above each player's cards."""
        # Define panel positions for each player (above their card area)
        if self.num_players == 4:
            panel_positions = {
                0: (
                    WINDOW_WIDTH // 2,
                    WINDOW_HEIGHT - 225,
                    "center",
                ),  # Above human hand
                1: (80, WINDOW_HEIGHT // 2 - 180, "center"),  # Above West's cards
                2: (WINDOW_WIDTH // 2, 10, "center"),  # Above North's cards
                3: (
                    WINDOW_WIDTH - 80,
                    WINDOW_HEIGHT // 2 - 180,
                    "center",
                ),  # Above East's cards
            }
        else:
            panel_positions = {
                0: (
                    WINDOW_WIDTH // 2,
                    WINDOW_HEIGHT - 225,
                    "center",
                ),  # Above human hand
                1: (150, WINDOW_HEIGHT // 2 - 210, "center"),  # Above Left's cards
                2: (
                    WINDOW_WIDTH - 150,
                    WINDOW_HEIGHT // 2 - 210,
                    "center",
                ),  # Above Right's cards
            }

        for i, player in enumerate(self.game.players):
            px, py, align = panel_positions[i]

            is_current = i == self.game.current_player_index

            # Render text
            name_color = GOLD if is_current else WHITE
            name_surf = self.label_font.render(player.name, True, name_color)

            score_text = f"Total: {player.total_score}  Round: {player.round_score}"
            score_surf = self.score_font.render(score_text, True, LIGHT_GRAY)

            # Calculate panel size
            panel_width = max(name_surf.get_width(), score_surf.get_width()) + 20
            panel_height = name_surf.get_height() + score_surf.get_height() + 15

            # Position panel based on alignment
            if align == "center":
                panel_x = px - panel_width // 2
            elif align == "left":
                panel_x = px
            else:  # right
                panel_x = px - panel_width

            panel_rect = pygame.Rect(panel_x, py, panel_width, panel_height)

            # Draw panel background
            bg_color = (0, 60, 0) if is_current else (0, 40, 0)
            pygame.draw.rect(self.screen, bg_color, panel_rect, border_radius=8)
            border_color = GOLD if is_current else (0, 80, 0)
            pygame.draw.rect(self.screen, border_color, panel_rect, 2, border_radius=8)

            # Draw name centered in panel
            name_rect = name_surf.get_rect(
                centerx=panel_rect.centerx, top=panel_rect.top + 5
            )
            self.screen.blit(name_surf, name_rect)

            # Draw score centered below name
            score_rect = score_surf.get_rect(
                centerx=panel_rect.centerx, top=name_rect.bottom + 3
            )
            self.screen.blit(score_surf, score_rect)

    def draw_human_hand(self):
        """Draw the human player's hand."""
        # In auto mode, all hands are drawn by draw_opponent_hands
        if self.auto_play:
            self.card_rects.clear()
            return

        player = self.game.players[self.human_player_index]
        hand = player.hand

        if not hand:
            return

        self.card_rects.clear()

        total_width = (len(hand) - 1) * CARD_SPACING + CARD_WIDTH
        start_x = (WINDOW_WIDTH - total_width) // 2
        y = WINDOW_HEIGHT - 140

        valid_plays = []
        if self.game.phase == GamePhase.PLAYING:
            valid_plays = self.game.get_valid_plays(self.human_player_index)

        for i, card in enumerate(hand):
            x = start_x + i * CARD_SPACING

            is_selected = card in self.selected_cards
            is_highlighted = card == self.hovered_card
            is_valid = card in valid_plays if valid_plays else True

            # Raise selected/hovered cards
            card_y = y
            if is_selected:
                card_y -= 20
            elif is_highlighted:
                card_y -= 10

            # Dim invalid cards during play phase
            card_surf = self.card_renderer.render_card(
                card, highlighted=is_highlighted and is_valid, selected=is_selected
            )

            if not is_valid and self.game.phase == GamePhase.PLAYING:
                # Dim the card
                dim_surf = pygame.Surface(card_surf.get_size(), pygame.SRCALPHA)
                dim_surf.fill((0, 0, 0, 100))
                card_surf = card_surf.copy()
                card_surf.blit(dim_surf, (0, 0))

            rect = pygame.Rect(x, card_y, CARD_WIDTH, CARD_HEIGHT)
            self.card_rects.append((rect, card))
            self.screen.blit(card_surf, (x, card_y))

    def draw_opponent_hands(self):
        """Draw opponent hands (face down, or face up in auto mode)."""
        for i, player in enumerate(self.game.players):
            if i == self.human_player_index and not self.auto_play:
                continue

            x, y, orientation = self.get_player_position(i)
            num_cards = len(player.hand)

            if num_cards == 0:
                continue

            # In auto mode, show all cards face up (small)
            show_face_up = self.auto_play

            if show_face_up:
                cards = sorted(player.hand, key=lambda c: (c.suit.value, c.rank.value))

                if orientation in ("left", "right"):
                    spacing = min(12, (180 // max(num_cards, 1)))
                    total_height = (num_cards - 1) * spacing + CARD_HEIGHT // 2
                    start_y = y - total_height // 2

                    for j, card in enumerate(cards):
                        card_y = start_y + j * spacing
                        card_x = x - CARD_WIDTH // 4
                        card_surf = self.card_renderer.render_card(card, small=True)
                        self.screen.blit(card_surf, (card_x, card_y))
                else:
                    spacing = min(18, (250 // max(num_cards, 1)))
                    total_width = (num_cards - 1) * spacing + CARD_WIDTH // 2
                    start_x = x - total_width // 2

                    for j, card in enumerate(cards):
                        card_x = start_x + j * spacing
                        card_surf = self.card_renderer.render_card(card, small=True)
                        self.screen.blit(card_surf, (card_x, y))
            else:
                card_back = self.card_renderer.render_card_back(small=True)

                if orientation in ("left", "right"):
                    # Vertical fan
                    spacing = min(15, (200 // max(num_cards, 1)))
                    total_height = (num_cards - 1) * spacing + card_back.get_height()
                    start_y = y - total_height // 2

                    for j in range(num_cards):
                        card_y = start_y + j * spacing
                        card_x = x - card_back.get_width() // 2
                        self.screen.blit(card_back, (card_x, card_y))
                else:
                    # Horizontal fan
                    spacing = min(15, (200 // max(num_cards, 1)))
                    total_width = (num_cards - 1) * spacing + card_back.get_width()
                    start_x = x - total_width // 2

                    for j in range(num_cards):
                        card_x = start_x + j * spacing
                        self.screen.blit(card_back, (card_x, y))

    def draw_current_trick(self):
        """Draw the cards played in the current trick."""
        if not self.played_cards_display:
            return

        center_x = WINDOW_WIDTH // 2
        center_y = WINDOW_HEIGHT // 2

        # Position offsets for each player's played card
        if self.num_players == 4:
            offsets = {
                0: (0, 40),
                1: (-60, 0),
                2: (0, -40),
                3: (60, 0),
            }
        else:
            offsets = {
                0: (0, 40),
                1: (-60, -20),
                2: (60, -20),
            }

        for player_idx, card in self.played_cards_display:
            offset = offsets[player_idx]
            x = center_x + offset[0] - CARD_WIDTH // 2
            y = center_y + offset[1] - CARD_HEIGHT // 2

            card_surf = self.card_renderer.render_card(card)
            self.screen.blit(card_surf, (x, y))

    def draw_pass_ui(self):
        """Draw the passing phase UI."""
        if self.game.phase != GamePhase.PASSING:
            return

        # Instructions
        direction = self.game.pass_direction.name.replace("_", " ").title()
        if self.auto_play:
            text = f"Passing {direction}..."
        else:
            text = f"Select 3 cards to pass {direction}"
        text_surf = self.label_font.render(text, True, WHITE)
        text_rect = text_surf.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 180))
        self.screen.blit(text_surf, text_rect)

        # Pass button (only in manual mode)
        if not self.auto_play:
            self.pass_button.enabled = len(self.selected_cards) == 3
            self.pass_button.draw(self.screen)

    def draw_message(self):
        """Draw temporary messages."""
        if self.message and pygame.time.get_ticks() < self.message_timer:
            text_surf = self.title_font.render(self.message, True, WHITE)
            text_rect = text_surf.get_rect(
                center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 150)
            )

            # Background
            bg_rect = text_rect.inflate(20, 10)
            pygame.draw.rect(self.screen, BLACK, bg_rect, border_radius=5)
            pygame.draw.rect(self.screen, WHITE, bg_rect, 2, border_radius=5)

            self.screen.blit(text_surf, text_rect)

    def draw_game_over(self):
        """Draw game over screen."""
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        # Winner announcement
        winner = self.game.get_winner()
        title = "Game Over!"
        title_surf = self.title_font.render(title, True, GOLD)
        title_rect = title_surf.get_rect(
            center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 80)
        )
        self.screen.blit(title_surf, title_rect)

        winner_text = f"{winner.name} wins with {winner.total_score} points!"
        winner_surf = self.label_font.render(winner_text, True, WHITE)
        winner_rect = winner_surf.get_rect(
            center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 180)
        )
        self.screen.blit(winner_surf, winner_rect)

        # Draw summary panel box
        box_width = 320
        box_height = self.num_players * 35 + 20
        box_rect = pygame.Rect(
            WINDOW_WIDTH // 2 - box_width // 2,
            WINDOW_HEIGHT // 2 - 150,
            box_width,
            box_height,
        )
        pygame.draw.rect(self.screen, (0, 40, 0, 200), box_rect, border_radius=10)
        pygame.draw.rect(self.screen, GOLD, box_rect, 2, border_radius=10)

        # Final scores
        y = box_rect.top + 25
        for player in sorted(self.game.players, key=lambda p: p.total_score):
            score_text = f"{player.name}: {player.total_score}"
            score_surf = self.label_font.render(score_text, True, WHITE)
            score_rect = score_surf.get_rect(center=(WINDOW_WIDTH // 2, y))
            self.screen.blit(score_surf, score_rect)
            y += 35

        if not self.auto_play:
            self.new_game_button.draw(self.screen)
        else:
            # Show "Starting new game..." message
            msg_surf = self.label_font.render("Starting new game...", True, LIGHT_GRAY)
            msg_rect = msg_surf.get_rect(center=(WINDOW_WIDTH // 2, y + 20))
            self.screen.blit(msg_surf, msg_rect)

    def draw_round_end(self):
        """Draw round end summary."""
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))

        title = f"Round {self.game.round_number} Complete"
        title_surf = self.title_font.render(title, True, GOLD)
        title_rect = title_surf.get_rect(
            center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 160)
        )
        self.screen.blit(title_surf, title_rect)

        # Round scores
        y = WINDOW_HEIGHT // 2 - 100
        for player in self.game.players:
            score_text = f"{player.name}: +{player.round_score} ({player.total_score})"
            score_surf = self.title_font.render(score_text, True, WHITE)
            score_rect = score_surf.get_rect(center=(WINDOW_WIDTH // 2, y))
            self.screen.blit(score_surf, score_rect)
            y += 45

        if not self.auto_play:
            self.continue_button.draw(self.screen)
        else:
            msg_surf = self.label_font.render(
                "Next round starting...", True, LIGHT_GRAY
            )
            msg_rect = msg_surf.get_rect(center=(WINDOW_WIDTH // 2, y + 10))
            self.screen.blit(msg_surf, msg_rect)

    def draw_kitty_info(self):
        """Draw kitty information for 3-player KITTY mode."""
        if (
            self.num_players != 3
            or self.three_player_mode != ThreePlayerMode.KITTY
            or self.game.kitty_card is None
        ):
            return

        if self.game.kitty_claimed:
            text = "Kitty claimed"
            color = GRAY
        else:
            text = "Kitty: unclaimed"
            color = GOLD

        text_surf = self.score_font.render(text, True, color)
        self.screen.blit(text_surf, (10, 10))

    def handle_card_click(self, pos: tuple[int, int]) -> Optional[Card]:
        """Check if a card was clicked and return it."""
        # Check in reverse order (top cards first)
        for rect, card in reversed(self.card_rects):
            if rect.collidepoint(pos):
                return card
        return None

    def handle_card_hover(self, pos: tuple[int, int]):
        """Update hovered card based on mouse position."""
        self.hovered_card = None
        for rect, card in reversed(self.card_rects):
            if rect.collidepoint(pos):
                self.hovered_card = card
                break

    def ai_pass_cards(self, player_index: int):
        """AI selects cards to pass using strategic heuristics."""
        player = self.game.players[player_index]
        ai = get_ai(player_index)
        cards_to_pass = ai.select_pass_cards(player.hand, self.game.pass_direction.name)
        self.game.set_pass_cards(player_index, cards_to_pass)

    def ai_play_card(self, player_index: int):
        """AI plays a card using strategic heuristics."""
        valid_plays = self.game.get_valid_plays(player_index)
        if not valid_plays:
            return

        player = self.game.players[player_index]
        ai = get_ai(player_index)

        card = ai.select_play(
            hand=player.hand,
            valid_plays=valid_plays,
            trick=self.game.current_trick,
            player_index=player_index,
            num_players=self.num_players,
            hearts_broken=self.game.hearts_broken,
        )

        result = self.game.play_card(player_index, card)
        self.played_cards_display.append((player_index, card))

        if result["trick_complete"]:
            # Record trick for all AI players
            record_trick_for_all(
                result["trick_cards"], result["trick_winner"], result["trick_info"]
            )

            self.display_timer = pygame.time.get_ticks() + 1000
            self.trick_winner_display = result["trick_winner"]
            winner_name = self.game.players[result["trick_winner"]].name
            self.set_message(f"{winner_name} takes the trick", 1000)

    def _auto_delay_elapsed(self) -> bool:
        """Check if enough time has passed since last auto action."""
        now = pygame.time.get_ticks()
        if now - self.last_auto_action >= self.play_delay_ms:
            self.last_auto_action = now
            return True
        return False

    def process_game_logic(self):
        """Process game logic, including AI turns."""
        if self.game.phase == GamePhase.PASSING:
            # In auto mode, all players use AI for passing
            if self.auto_play:
                for i in range(self.num_players):
                    if len(self.game.players[i].cards_to_pass) < 3:
                        self.ai_pass_cards(i)
                # Auto-execute pass after delay
                if self.game.all_players_ready_to_pass() and self._auto_delay_elapsed():
                    self.game.execute_pass()
                    self.set_message("Cards passed!")
            else:
                # AI passes cards (non-human players)
                for i in range(self.num_players):
                    if i != self.human_player_index:
                        if len(self.game.players[i].cards_to_pass) < 3:
                            self.ai_pass_cards(i)

        elif self.game.phase == GamePhase.PLAYING:
            # Check if displaying trick result
            if self.display_timer > 0:
                if pygame.time.get_ticks() >= self.display_timer:
                    self.display_timer = 0
                    self.played_cards_display.clear()
                    self.trick_winner_display = None
                return

            # In auto mode, all players use AI
            if self.auto_play:
                if self._auto_delay_elapsed():
                    current = self.game.current_player_index
                    self.ai_play_card(current)
            else:
                # AI turn (non-human players)
                current = self.game.current_player_index
                if current != self.human_player_index:
                    self.ai_play_card(current)

        elif self.game.phase == GamePhase.ROUND_END:
            # In auto mode, automatically continue to next round
            if self.auto_play and self._auto_delay_elapsed():
                self.game.start_round()
                self.selected_cards.clear()
                self.played_cards_display.clear()
                # Reset AI for new round
                reset_all_ai(self.num_players, mode=self.game.game_mode)

        elif self.game.phase == GamePhase.GAME_OVER:
            # In auto mode, automatically start new game
            if self.auto_play and self._auto_delay_elapsed():
                self.start_new_game()

    def handle_event(self, event: pygame.event.Event):
        """Handle pygame events."""
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEMOTION:
            self.handle_card_hover(event.pos)
            self.pass_button.handle_event(event)
            self.new_game_button.handle_event(event)
            self.continue_button.handle_event(event)

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Game over screen
            if self.game.phase == GamePhase.GAME_OVER:
                if self.new_game_button.handle_event(event):
                    self.start_new_game()
                return

            # Round end screen
            if self.game.phase == GamePhase.ROUND_END:
                if self.continue_button.handle_event(event):
                    self.game.start_round()
                    self.selected_cards.clear()
                    self.played_cards_display.clear()
                    # Reset AI for new round
                    reset_all_ai(self.num_players, mode=self.game.game_mode)
                return

            # Passing phase
            if self.game.phase == GamePhase.PASSING:
                if self.pass_button.handle_event(event):
                    if len(self.selected_cards) == 3:
                        self.game.set_pass_cards(
                            self.human_player_index, self.selected_cards
                        )
                        self.game.execute_pass()
                        self.selected_cards.clear()
                        self.set_message("Cards passed!")
                    return

                # Card selection
                card = self.handle_card_click(event.pos)
                if card:
                    if card in self.selected_cards:
                        self.selected_cards.remove(card)
                    elif len(self.selected_cards) < 3:
                        self.selected_cards.append(card)

            # Playing phase
            elif self.game.phase == GamePhase.PLAYING:
                if self.display_timer > 0:
                    return  # Wait for trick display

                if self.game.current_player_index == self.human_player_index:
                    card = self.handle_card_click(event.pos)
                    if card:
                        valid_plays = self.game.get_valid_plays(self.human_player_index)
                        if card in valid_plays:
                            result = self.game.play_card(self.human_player_index, card)
                            self.played_cards_display.append(
                                (self.human_player_index, card)
                            )

                            if result["trick_complete"]:
                                # Record trick for all AI players
                                record_trick_for_all(
                                    result["trick_cards"],
                                    result["trick_winner"],
                                    result["trick_info"],
                                )

                                self.display_timer = pygame.time.get_ticks() + 1000
                                self.trick_winner_display = result["trick_winner"]
                                winner_name = self.game.players[
                                    result["trick_winner"]
                                ].name
                                self.set_message(f"{winner_name} takes the trick", 1000)

    def draw_auto_indicator(self):
        """Draw auto-play mode indicator."""
        if not self.auto_play:
            return

        text = f"AUTO MODE (delay: {self.play_delay_ms / 1000:.1f}s)"
        text_surf = self.score_font.render(text, True, GOLD)
        text_rect = text_surf.get_rect(topright=(WINDOW_WIDTH - 10, 10))

        # Background
        bg_rect = text_rect.inflate(10, 4)
        pygame.draw.rect(self.screen, (0, 40, 0), bg_rect, border_radius=4)
        pygame.draw.rect(self.screen, GOLD, bg_rect, 1, border_radius=4)

        self.screen.blit(text_surf, text_rect)

    def draw(self):
        """Draw the game."""
        self.draw_background()
        self.draw_player_info()
        self.draw_opponent_hands()
        self.draw_current_trick()
        self.draw_human_hand()
        self.draw_kitty_info()
        self.draw_auto_indicator()

        if self.game.phase == GamePhase.PASSING:
            self.draw_pass_ui()
        elif self.game.phase == GamePhase.GAME_OVER:
            self.draw_game_over()
        elif self.game.phase == GamePhase.ROUND_END:
            self.draw_round_end()

        self.draw_message()

        pygame.display.flip()

    def run(self):
        """Main game loop."""
        while True:
            for event in pygame.event.get():
                self.handle_event(event)

            self.process_game_logic()
            self.draw()
            self.clock.tick(60)


def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Hearts Card Game")
    parser.add_argument(
        "--players",
        "-p",
        type=int,
        choices=[3, 4],
        default=4,
        help="Number of players (3 or 4)",
    )
    parser.add_argument(
        "--kitty", action="store_true", help="Use kitty mode for 3-player game"
    )
    parser.add_argument(
        "--auto", action="store_true", help="All players are AI (watch mode)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between AI plays in seconds (default: 1.0)",
    )

    args = parser.parse_args()

    three_player_mode = ThreePlayerMode.REMOVE_CARD
    if args.players == 3 and args.kitty:
        three_player_mode = ThreePlayerMode.KITTY

    gui = HeartsGUI(
        num_players=args.players,
        three_player_mode=three_player_mode,
        auto_play=args.auto,
        play_delay=args.delay,
    )
    gui.run()


if __name__ == "__main__":
    main()
