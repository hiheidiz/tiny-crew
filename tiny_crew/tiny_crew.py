"""
Tiny Crew: A simplified version of The Crew trick-taking cooperative game.

This is a simplified implementation for OpenSpiel with:
- 3 players (fixed)
- Simplified deck: color cards 1-4 in 4 colors + 3 submarines
- 1-2 tasks per mission
- Cooperative reward structure
"""

import pyspiel
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import random


# Game constants - SIMPLIFIED for faster CFR
NUM_COLORS = 3  # 3 colors (balanced size)
COLOR_VALUES = [1, 2, 3]  # 3 values per color
SUBMARINE_VALUES = [1, 2, 3]
COLORS = ['red', 'blue', 'green']  # 3 colors


class Card:
    """Represents a playing card."""
    
    def __init__(self, color: Optional[str], value: int, is_submarine: bool = False):
        self.color = color  # None for submarine cards
        self.value = value
        self.is_submarine = is_submarine
    
    def __repr__(self):
        if self.is_submarine:
            return f"Submarine({self.value})"
        return f"{self.color}({self.value})"
    
    def __eq__(self, other):
        return (self.color == other.color and 
                self.value == other.value and 
                self.is_submarine == other.is_submarine)
    
    def __hash__(self):
        return hash((self.color, self.value, self.is_submarine))


class Task:
    """Represents a task card."""
    
    def __init__(self, task_type: str, **kwargs):
        self.task_type = task_type
        self.completed = False
        self.assigned_player = None
        self.params = kwargs  # Task-specific parameters
    
    def __repr__(self):
        return f"Task({self.task_type}, {self.params})"


def create_deck() -> List[Card]:
    """Create a simplified deck for 3 players."""
    deck = []
    
    # Add color cards: 1-3 in each of 3 colors = 9 cards
    for color in COLORS:
        for value in COLOR_VALUES:
            deck.append(Card(color, value, is_submarine=False))
    
    # Add 3 submarine cards (one per player)
    for value in SUBMARINE_VALUES:
        deck.append(Card(None, value, is_submarine=True))
    
    return deck


def create_simple_tasks() -> List[Task]:
    """Create simplified tasks for the mission (3 players)."""
    tasks = []
    
    # Task 1: win a trick with highest red card
    tasks.append(Task("win_highest", color="red"))
    
    # Task 2: win exactly 1 blue card
    tasks.append(Task("win_exactly", color="blue", count=1))
    
    return tasks


class TinyCrewState(pyspiel.State):
    """State representation for Tiny Crew game."""
    
    def __init__(self, game, initialize=True):
        super().__init__(game)
        self._hands: Dict[int, List[Card]] = {}
        self._current_trick: List[Tuple[int, Card]] = []  # (player, card)
        self._trick_history: List[List[Tuple[int, Card]]] = []  # History of completed tricks
        self._tasks: List[Task] = []
        self._task_assignment: Dict[int, List[Task]] = defaultdict(list)
        self._current_player = 0
        self._trick_leader = None  # Player who leads the current trick
        self._leading_suit = None  # Suit of the first card in current trick
        self._num_tricks_won: Dict[int, int] = defaultdict(int)
        self._cards_won: Dict[int, List[Card]] = defaultdict(list)
        self._communication_used: Set[int] = set()  # Players who have communicated
        
        # Initialize game only if requested (not when cloning)
        if initialize:
            self._deal_cards()
            self._assign_tasks()
    
    def _deal_cards(self):
        """Deal cards to all players."""
        deck = create_deck()
        random.shuffle(deck)
        
        num_players = 3  # Fixed at 3 players
        cards_per_player = len(deck) // num_players
        
        for player in range(num_players):
            start_idx = player * cards_per_player
            end_idx = start_idx + cards_per_player
            self._hands[player] = deck[start_idx:end_idx]
    
    def _assign_tasks(self):
        """Assign tasks to players."""
        self._tasks = create_simple_tasks()
        
        # Simple assignment: assign tasks sequentially to 3 players
        for i, task in enumerate(self._tasks):
            player = i % 3
            task.assigned_player = player
            self._task_assignment[player].append(task)
    
    def current_player(self) -> int:
        """Returns the current player."""
        if self.is_terminal():
            return pyspiel.PlayerId.TERMINAL
        
        # If trick is complete, winner leads next trick
        if len(self._current_trick) == self.get_game().num_players():
            return pyspiel.PlayerId.TERMINAL  # Will be handled in apply_action
        
        # If trick just started, return trick leader
        if self._trick_leader is not None and len(self._current_trick) == 0:
            return self._trick_leader
        
        # Otherwise, next player in trick
        if len(self._current_trick) == 0:
            return 0  # First trick, player 0 leads
        else:
            return (self._current_trick[-1][0] + 1) % self.get_game().num_players()
    
    def legal_actions(self, player: Optional[int] = None) -> List[int]:
        """Returns legal actions for the current player."""
        if self.is_terminal():
            return []
        
        if player is None:
            player = self.current_player()
        
        if player == pyspiel.PlayerId.TERMINAL:
            return []
        
        legal_cards = []
        
        # If this is the first card in a trick, any card can be played
        if len(self._current_trick) == 0:
            legal_cards = self._hands[player]
        else:
            # Must follow suit if possible
            leading_card = self._current_trick[0][1]
            
            if leading_card.is_submarine:
                # Must follow submarine if possible
                submarine_cards = [c for c in self._hands[player] if c.is_submarine]
                if submarine_cards:
                    legal_cards = submarine_cards
                else:
                    legal_cards = self._hands[player]  # Can play any card
            else:
                # Must follow color if possible
                color_cards = [c for c in self._hands[player] 
                             if c.color == leading_card.color and not c.is_submarine]
                if color_cards:
                    legal_cards = color_cards
                else:
                    legal_cards = self._hands[player]  # Can play any card
        
        # Convert cards to action indices
        # Action encoding: (color_idx * len(COLOR_VALUES) + value - 1) for color cards
        # or (num_color_cards + value - 1) for submarines
        num_color_cards = len(COLORS) * len(COLOR_VALUES)
        actions = []
        for card in legal_cards:
            if card.is_submarine:
                action = num_color_cards + card.value - 1
            else:
                color_idx = COLORS.index(card.color)
                action = color_idx * len(COLOR_VALUES) + card.value - 1
            actions.append(action)
        
        return sorted(set(actions))
    
    def _action_to_card(self, action: int) -> Card:
        """Convert action index to Card object."""
        num_color_cards = len(COLORS) * len(COLOR_VALUES)
        if action < num_color_cards:
            # Color card
            color_idx = action // len(COLOR_VALUES)
            value = (action % len(COLOR_VALUES)) + 1
            return Card(COLORS[color_idx], value, is_submarine=False)
        else:
            # Submarine card
            value = (action - num_color_cards) + 1
            return Card(None, value, is_submarine=True)
    
    def apply_action(self, action: int):
        """Apply an action (play a card)."""
        player = self.current_player()
        card = self._action_to_card(action)
        
        # Remove card from hand
        self._hands[player].remove(card)
        
        # Add to current trick
        if len(self._current_trick) == 0:
            self._trick_leader = player
            if card.is_submarine:
                self._leading_suit = "submarine"
            else:
                self._leading_suit = card.color
        
        self._current_trick.append((player, card))
        
        # Check if trick is complete
        if len(self._current_trick) == self.get_game().num_players():
            self._resolve_trick()
    
    def clone(self):
        """Create a deep copy of the state."""
        import copy
        cloned = TinyCrewState(self.get_game(), initialize=False)
        
        # Deep copy all state variables
        cloned._hands = {p: hand.copy() for p, hand in self._hands.items()}
        cloned._current_trick = self._current_trick.copy()
        cloned._trick_history = [trick.copy() for trick in self._trick_history]
        cloned._tasks = copy.deepcopy(self._tasks)
        # Copy task_assignment - ensure all players are represented
        cloned._task_assignment = defaultdict(list)
        for p, tasks in self._task_assignment.items():
            cloned._task_assignment[p] = tasks.copy()
        cloned._current_player = self._current_player
        cloned._trick_leader = self._trick_leader
        cloned._leading_suit = self._leading_suit
        cloned._num_tricks_won = self._num_tricks_won.copy()
        cloned._cards_won = {p: cards.copy() for p, cards in self._cards_won.items()}
        cloned._communication_used = self._communication_used.copy()
        
        return cloned
    
    def child(self, action: int):
        """Return a new state after applying an action (for OpenSpiel compatibility)."""
        new_state = self.clone()
        new_state.apply_action(action)
        return new_state
    
    def _resolve_trick(self):
        """Resolve the current trick and determine winner."""
        # Find winner: highest card of leading suit, or highest submarine
        leading_card = self._current_trick[0][1]
        winner_idx = 0
        winning_card = leading_card
        
        if leading_card.is_submarine:
            # Submarine trick: highest submarine wins
            for i, (player, card) in enumerate(self._current_trick):
                if card.is_submarine and card.value > winning_card.value:
                    winner_idx = i
                    winning_card = card
        else:
            # Color trick: highest card of leading color wins
            # Submarines can win if no submarine was led
            for i, (player, card) in enumerate(self._current_trick):
                if card.is_submarine:
                    # Submarine trumps color cards
                    if not winning_card.is_submarine:
                        winner_idx = i
                        winning_card = card
                elif card.color == leading_card.color:
                    # Same color: compare values
                    if card.value > winning_card.value or winning_card.is_submarine:
                        winner_idx = i
                        winning_card = card
        
        winner_player = self._current_trick[winner_idx][0]
        
        # Update trick history
        self._trick_history.append(self._current_trick.copy())
        
        # Update player stats
        self._num_tricks_won[winner_player] += 1
        # Ensure defaultdict behavior
        if winner_player not in self._cards_won:
            self._cards_won[winner_player] = []
        for _, card in self._current_trick:
            self._cards_won[winner_player].append(card)
        
        # Check task completion
        self._check_tasks()
        
        # Reset for next trick
        self._current_trick = []
        self._trick_leader = winner_player
        self._leading_suit = None
    
    def _check_tasks(self):
        """Check if any tasks have been completed."""
        for task in self._tasks:
            if task.completed:
                continue
            
            player = task.assigned_player
            
            if task.task_type == "win_highest":
                # Check if player won a trick with highest card of specified color
                color = task.params["color"]
                max_value = max(COLOR_VALUES)  # Highest value in simplified deck
                for trick in self._trick_history:
                    if trick[-1][0] == player:  # Player won this trick
                        for _, card in trick:
                            if card.color == color and card.value == max_value:
                                task.completed = True
                                break
            
            elif task.task_type == "win_exactly":
                # Check if player won exactly N cards of specified color
                color = task.params["color"]
                count = task.params["count"]
                # Handle case where player hasn't won any cards yet
                cards_won = self._cards_won.get(player, [])
                color_cards_won = sum(1 for card in cards_won if card.color == color)
                if color_cards_won == count and len(self._hands.get(player, [])) == 0:
                    # All cards played and exact count achieved
                    task.completed = True
    
    def is_terminal(self) -> bool:
        """Check if game is over."""
        # Game ends when all cards are played
        if all(len(hand) == 0 for hand in self._hands.values() if hand):
            return True
        
        # Game ends if all tasks completed
        if all(task.completed for task in self._tasks):
            return True
        
        # Game ends if task becomes impossible (simplified check)
        # In full implementation, would check more carefully
        return False
    
    def returns(self) -> List[float]:
        """Returns rewards for each player."""
        if not self.is_terminal():
            return [0.0] * self.get_game().num_players()
        
        # Cooperative reward: +1 if all tasks completed, 0 otherwise
        all_completed = all(task.completed for task in self._tasks)
        reward = 1.0 if all_completed else 0.0
        
        return [reward] * self.get_game().num_players()
    
    def information_state_string(self, player: int) -> str:
        """Returns information state string for a player."""
        parts = []
        
        # Player's hand
        hand_str = ",".join(str(card) for card in sorted(self._hands.get(player, []), 
                                                         key=lambda c: (c.color or "z", c.value)))
        parts.append(f"hand:{hand_str}")
        
        # Assigned tasks (handle case where player might not have tasks)
        task_strs = [f"{t.task_type}({t.params})" for t in self._task_assignment.get(player, [])]
        parts.append(f"tasks:{','.join(task_strs)}")
        
        # Current trick (all players can see)
        # Keep player positions - needed for suit-following rules
        if self._current_trick:
            trick_str = ",".join(f"P{p}:{c}" for p, c in self._current_trick)
            parts.append(f"trick:{trick_str}")
        
        # Previous tricks: only track who won (for turn order)
        # This reduces info state explosion while preserving essential information
        if self._trick_leader is not None and len(self._current_trick) == 0:
            # Trick leader indicates who won the last trick
            parts.append(f"last_winner:P{self._trick_leader}")
        
        return "|".join(parts)
    
    def __str__(self):
        """String representation of state."""
        lines = []
        lines.append("Tiny Crew Game State:")
        lines.append(f"Current Player: {self.current_player()}")
        lines.append(f"Trick Leader: {self._trick_leader}")
        
        for player in range(self.get_game().num_players()):
            hand_str = ", ".join(str(c) for c in self._hands[player])
            tasks_str = ", ".join(str(t) for t in self._task_assignment[player])
            lines.append(f"Player {player}: Hand=[{hand_str}], Tasks=[{tasks_str}]")
        
        if self._current_trick:
            trick_str = ", ".join(f"P{p}:{c}" for p, c in self._current_trick)
            lines.append(f"Current Trick: [{trick_str}]")
        
        return "\n".join(lines)


class TinyCrewGame(pyspiel.Game):
    """Tiny Crew game definition."""
    
    def __init__(self, params: Optional[Dict] = None):
        if params is None:
            params = {}
        
        num_players = 3  # Fixed at 3 players
        
        game_info = pyspiel.GameInfo(
            num_distinct_actions=12,  # 9 color cards (3 colors * 3 values) + 3 submarines
            max_chance_outcomes=0,
            num_players=num_players,
            min_utility=0.0,
            max_utility=1.0,
            utility_sum=num_players * 1.0,  # Cooperative: all get same reward
            max_game_length=20  # Upper bound (12 cards total)
        )
        
        super().__init__(pyspiel.GameType(
            short_name="tiny_crew",
            long_name="Tiny Crew",
            dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
            chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
            information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
            utility=pyspiel.GameType.Utility.IDENTICAL,  # Cooperative
            reward_model=pyspiel.GameType.RewardModel.TERMINAL,
            max_num_players=3,
            min_num_players=3,
            provides_information_state_string=True,
            provides_information_state_tensor=False,
            provides_observation_string=False,
            provides_observation_tensor=False,
            parameter_specification={}
        ), game_info, params)
        
        self._num_players = num_players
    
    def num_players(self) -> int:
        return self._num_players
    
    def new_initial_state(self) -> TinyCrewState:
        return TinyCrewState(self)
    
    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns a default observer."""
        return pyspiel.InfoStateObserver(self)


# Note: Full OpenSpiel registration requires C++ bindings.
# For Python-only testing, the game can be instantiated directly:
if __name__ == "__main__":
    # Test the game
    game = TinyCrewGame()
    state = game.new_initial_state()
    
    print("Initial State:")
    print(state)
    print("\n" + "="*60 + "\n")
    
    # Play a few actions
    for i in range(6):  # Play 2 tricks (3 players * 2 tricks)
        if state.is_terminal():
            break
        
        player = state.current_player()
        legal_actions = state.legal_actions()
        
        if legal_actions:
            action = legal_actions[0]  # Play first legal action
            print(f"Player {player} plays action {action}")
            state.apply_action(action)
            print(f"After action: {state.information_state_string(player)}")
            print()
    
    print("Final State:")
    print(state)
    print(f"\nReturns: {state.returns()}")

