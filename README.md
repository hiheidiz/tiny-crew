# Tiny Crew Game Implementation

A simplified version of "The Crew: Mission Deep Sea" trick-taking cooperative game implemented for OpenSpiel.

## Overview

Tiny Crew is a cooperative trick-taking card game where players work together to complete tasks by winning specific tricks. This implementation simplifies the original game for use in multi-agent learning research.

## File Structure

```
.
├── README.md              # This file
├── tiny_crew.py          # Main game implementation
├── tiny_crew_cfr.py      # CFR solver with policy evaluation
├── test_tiny_crew.py     # Unit tests
└── __init__.py           # Package initialization
```

## Game Rules

### Setup
- **Players**: 3 players (fixed)
- **Deck**: 
  - Color cards: values 1-3 in 3 colors (red, blue, green) = 9 cards
  - Submarine cards: 3 submarines (values 1, 2, 3) = 3 cards
  - Total: 12 cards
- **Dealing**: All cards are dealt to players (no leftover cards)
- **Tasks**: 2 simplified tasks per mission

### Gameplay

1. **Task Assignment**: Tasks are assigned to players at the start
   - Example tasks:
     - "Win a trick with the highest red card"
     - "Win exactly 1 blue card"

2. **Trick-Taking**:
   - Players take turns playing one card
   - First player leads the trick (captain leads first trick)
   - Subsequent tricks are led by the winner of the previous trick
   - **Follow Suit Rule**: Must play a card of the same color as the lead card if possible
   - If no card of that color, any card may be played
   - **Submarine Cards**: Act as trump cards and always win (unless multiple submarines, then highest wins)

3. **Trick Resolution**:
   - Winner: Highest card of the leading suit
   - Exception: Submarine cards trump color cards
   - Winner takes all cards from the trick

4. **Task Completion**:
   - Tasks are checked after each trick
   - A task is completed when its conditions are met
   - Example: "Win highest red card" completes when player wins a trick containing red 3 (highest value)

5. **Game End**:
   - **Win**: All tasks completed → All players receive +1 reward
   - **Loss**: Game ends without completing all tasks → All players receive 0 reward

### Information Sets

Each player observes:
- Their own hand
- Tasks assigned to them
- Current trick (all cards played so far)
- Last trick winner (for turn order)

## Implementation Details

### Key Classes

1. **`Card`**: Represents a playing card
   - Color (red, blue, green) or None for submarines
   - Value (1-3)
   - Submarine flag

2. **`Task`**: Represents a task card
   - Task type (e.g., "win_highest", "win_exactly")
   - Parameters (color, count, etc.)
   - Completion status
   - Assigned player

3. **`TinyCrewState`**: Game state
   - Player hands
   - Current trick
   - Trick history
   - Task assignments
   - Task completion status

4. **`TinyCrewGame`**: Game definition for OpenSpiel
   - Game parameters
   - Game info (players, actions, utilities)
   - State creation

### Action Encoding

Actions are encoded as integers:
- Color cards: `color_idx * 3 + value - 1` (0-8)
  - Red: 0-2, Blue: 3-5, Green: 6-8
- Submarine cards: `9 + value - 1` (9-11)

### Information State String Format

```
hand:Card1,Card2,...|tasks:task1,task2,...|trick:P0:Card,P1:Card,...|last_winner:P0
```

## Usage

### Basic Usage

```python
from tiny_crew import TinyCrewGame

# Create game (always 3 players)
game = TinyCrewGame()
state = game.new_initial_state()

# Play actions
while not state.is_terminal():
    player = state.current_player()
    legal_actions = state.legal_actions()
    
    # Choose action (e.g., first legal action)
    action = legal_actions[0]
    state.apply_action(action)

# Get rewards
rewards = state.returns()  # [1.0, 1.0, 1.0] if all tasks completed
```

### Running CFR

Train a CFR policy:

```bash
cd tiny_crew
python3 tiny_crew_cfr.py 5000
```

This will:
1. Run 5000 iterations of chance-sampled CFR
2. Evaluate the learned policy
3. Compare against random baseline
4. Analyze policy quality

### Testing

Run the test suite:
```bash
cd tiny_crew
python3 test_tiny_crew.py
```

Tests cover:
- Card and deck creation
- Task creation and assignment
- Game initialization
- Legal actions
- Trick resolution
- Follow suit rules
- Full game play
- Information state encoding

## CFR Solver

The `tiny_crew_cfr.py` file implements a chance-sampled CFR solver with:

- **Chance Sampling**: Samples a new deal each iteration
- **Lazy Initialization**: Creates info state nodes as needed
- **Policy Evaluation**: Automatically evaluates learned policies
- **Baseline Comparison**: Compares against random play

### Evaluation Metrics

After training, the solver reports:
- **Win Rate**: Percentage of games where all tasks are completed
- **Average Reward**: Expected reward per game
- **Policy Quality**: Analysis of deterministic vs. uniform states
- **Regret Convergence**: How well regrets have converged

## Simplifications from Original Game

1. **Fixed player count**: Always 3 players (instead of 3-5 configurable)
2. **Reduced deck size**: Only values 1-3 instead of 1-9
3. **Fewer colors**: 3 colors instead of 4
4. **Fewer tasks**: 2 tasks instead of multiple complex tasks
5. **No communication**: Simplified (communication can be added later)
6. **Simple task types**: Only basic task types implemented
7. **Fixed task assignment**: Sequential assignment instead of captain-based selection

## Future Enhancements

- [ ] Add communication/hinting mechanism
- [ ] More complex task types
- [ ] Captain-based task selection
- [ ] Distress signal mechanic
- [ ] Better task impossibility detection
- [ ] Observation tensor encoding
- [ ] Integration with OpenSpiel C++ bindings for full registration
- [ ] External sampling MCCFR variant
- [ ] Policy visualization tools

## References

- Original game: "The Crew: Mission Deep Sea" by Thomas Sing
- OpenSpiel: https://github.com/deepmind/open_spiel
- CFR Algorithm: Counterfactual Regret Minimization for extensive-form games

