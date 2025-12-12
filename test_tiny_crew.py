"""
Test file for Tiny Crew game implementation.
"""

import sys
import os
# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tiny_crew import (
    TinyCrewGame, TinyCrewState, Card, create_deck, create_simple_tasks,
    NUM_COLORS, COLOR_VALUES, COLORS
)
import numpy as np


def test_card_creation():
    """Test Card creation."""
    print("Testing Card creation...")
    card1 = Card("red", 4, is_submarine=False)
    card2 = Card(None, 2, is_submarine=True)
    print(f"  Color card: {card1}")
    print(f"  Submarine card: {card2}")
    assert card1.color == "red"
    assert card1.value == 4
    assert not card1.is_submarine
    assert card2.is_submarine
    print("  ✓ Card creation works\n")


def test_deck_creation():
    """Test deck creation."""
    print("Testing deck creation...")
    deck = create_deck()
    print(f"  3 players: {len(deck)} cards")
    
    # Check card counts
    color_cards = [c for c in deck if not c.is_submarine]
    submarine_cards = [c for c in deck if c.is_submarine]
    
    assert len(color_cards) == NUM_COLORS * len(COLOR_VALUES), \
        f"Expected {NUM_COLORS * len(COLOR_VALUES)} color cards"
    assert len(submarine_cards) == 3, \
        f"Expected 3 submarine cards"
    
    print(f"    Color cards: {len(color_cards)}, Submarines: {len(submarine_cards)}")
    print("  ✓ Deck creation works\n")


def test_task_creation():
    """Test task creation."""
    print("Testing task creation...")
    tasks = create_simple_tasks()
    print(f"  3 players: {len(tasks)} tasks")
    for task in tasks:
        print(f"    {task}")
    print("  ✓ Task creation works\n")


def test_game_initialization():
    """Test game initialization."""
    print("Testing game initialization...")
    game = TinyCrewGame()
    state = game.new_initial_state()
    
    print(f"  Players: {game.num_players()}")
    assert game.num_players() == 3, "Should have exactly 3 players"
    print(f"  Initial player: {state.current_player()}")
    print(f"  Is terminal: {state.is_terminal()}")
    
    # Check hands are dealt
    for player in range(3):
        hand_size = len(state._hands[player])
        print(f"  Player {player} hand size: {hand_size}")
        assert hand_size > 0, "Hand should not be empty"
    
    # Check tasks are assigned
    assert len(state._tasks) == 2, "Should have 2 tasks"
    print(f"  Tasks: {len(state._tasks)}")
    
    print("  ✓ Game initialization works\n")


def test_legal_actions():
    """Test legal actions."""
    print("Testing legal actions...")
    game = TinyCrewGame()
    state = game.new_initial_state()
    
    player = state.current_player()
    legal_actions = state.legal_actions()
    
    print(f"  Player {player} legal actions: {len(legal_actions)}")
    assert len(legal_actions) > 0, "Should have legal actions"
    assert len(legal_actions) == len(state._hands[player]), \
        "First player should be able to play any card"
    
    print("  ✓ Legal actions work\n")


def test_trick_resolution():
    """Test trick resolution."""
    print("Testing trick resolution...")
    game = TinyCrewGame()
    state = game.new_initial_state()
    
    # Play a complete trick
    print("  Playing a trick...")
    for i in range(game.num_players()):
        if state.is_terminal():
            break
        
        player = state.current_player()
        legal_actions = state.legal_actions()
        
        if legal_actions:
            # Play first legal action
            action = legal_actions[0]
            card = state._action_to_card(action)
            print(f"    Player {player} plays {card}")
            state.apply_action(action)
    
    print(f"  Trick history length: {len(state._trick_history)}")
    assert len(state._trick_history) == 1, "Should have completed one trick"
    
    # Check that a player won
    tricks_won = sum(state._num_tricks_won.values())
    assert tricks_won == 1, "One player should have won the trick"
    
    print("  ✓ Trick resolution works\n")


def test_follow_suit_rule():
    """Test that players must follow suit when possible."""
    print("Testing follow suit rule...")
    game = TinyCrewGame()
    state = game.new_initial_state()
    
    # Player 0 leads with a red card
    player0 = 0
    legal_actions = state.legal_actions(player0)
    action = legal_actions[0]
    card0 = state._action_to_card(action)
    
    # Find a red card if possible
    for action in legal_actions:
        card = state._action_to_card(action)
        if card.color == "red" and not card.is_submarine:
            action = action
            card0 = card
            break
    
    print(f"  Player 0 leads with {card0}")
    state.apply_action(action)
    
    # Player 1 must follow suit if they have red
    player1 = state.current_player()
    player1_hand = state._hands[player1]
    has_red = any(c.color == "red" and not c.is_submarine for c in player1_hand)
    
    legal_actions_p1 = state.legal_actions(player1)
    legal_cards_p1 = [state._action_to_card(a) for a in legal_actions_p1]
    
    if has_red:
        # Should only be able to play red cards
        all_red = all(c.color == "red" and not c.is_submarine for c in legal_cards_p1)
        print(f"  Player 1 has red cards: {has_red}")
        print(f"  Legal actions are all red: {all_red}")
        assert all_red or any(c.is_submarine for c in legal_cards_p1), \
            "Should only be able to play red cards if has red"
    else:
        print(f"  Player 1 has no red cards, can play any card")
        assert len(legal_actions_p1) == len(player1_hand), \
            "Should be able to play any card if no red"
    
    print("  ✓ Follow suit rule works\n")


def test_game_play():
    """Test full game play."""
    print("Testing full game play...")
    game = TinyCrewGame()
    state = game.new_initial_state()
    
    max_actions = 100
    action_count = 0
    
    while not state.is_terminal() and action_count < max_actions:
        player = state.current_player()
        legal_actions = state.legal_actions()
        
        if not legal_actions:
            break
        
        # Play random legal action
        action = np.random.choice(legal_actions)
        state.apply_action(action)
        action_count += 1
    
    print(f"  Played {action_count} actions")
    print(f"  Is terminal: {state.is_terminal()}")
    print(f"  Returns: {state.returns()}")
    print(f"  Tricks won: {dict(state._num_tricks_won)}")
    
    assert state.is_terminal(), "Game should have ended"
    
    print("  ✓ Full game play works\n")


def test_information_state():
    """Test information state string."""
    print("Testing information state...")
    game = TinyCrewGame()
    state = game.new_initial_state()
    
    for player in range(3):
        info_state = state.information_state_string(player)
        print(f"  Player {player} info state length: {len(info_state)}")
        assert len(info_state) > 0, "Info state should not be empty"
        assert "hand:" in info_state, "Info state should contain hand"
        assert "tasks:" in info_state, "Info state should contain tasks"
    
    print("  ✓ Information state works\n")


if __name__ == "__main__":
    print("="*60)
    print("Testing Tiny Crew Game Implementation")
    print("="*60 + "\n")
    
    try:
        test_card_creation()
        test_deck_creation()
        test_task_creation()
        test_game_initialization()
        test_legal_actions()
        test_trick_resolution()
        test_follow_suit_rule()
        test_game_play()
        test_information_state()
        
        print("="*60)
        print("All tests passed! ✓")
        print("="*60)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

