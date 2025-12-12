"""
Correct CFR implementation for Tiny Crew based on OpenSpiel's CFR algorithm.

This implementation correctly handles:
- Reach probability tracking (separate for each player)
- Counterfactual utility computation
- Regret updates
- Average policy computation
"""

import sys
import os
# Add parent directory to path to import tiny_crew
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tiny_crew.tiny_crew import TinyCrewGame
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple


class InfoStateNode:
    """Node storing regrets and cumulative policy for an information state."""
    def __init__(self, legal_actions: List[int]):
        self.legal_actions = legal_actions
        self.cumulative_regret: Dict[int, float] = defaultdict(float)
        self.cumulative_policy: Dict[int, float] = defaultdict(float)


def regret_matching(cumulative_regrets: Dict[int, float], legal_actions: List[int]) -> Dict[int, float]:
    """Compute strategy using regret matching."""
    positive_regrets = {a: max(0.0, cumulative_regrets.get(a, 0.0)) for a in legal_actions}
    sum_positive = sum(positive_regrets.values())
    
    if sum_positive > 0:
        return {a: positive_regrets[a] / sum_positive for a in legal_actions}
    else:
        uniform_prob = 1.0 / len(legal_actions) if legal_actions else 1.0
        return {a: uniform_prob for a in legal_actions}


class CFRSolver:
    """
    Chance-Sampled CFR Solver for Tiny Crew.
    Generates a new random deal (Chance Sampling) every iteration.
    """
    
    def __init__(self, game: TinyCrewGame):
        self.game = game
        self.num_players = game.num_players()
        
        # Store info state nodes (regrets/policy)
        # We cannot pre-initialize because the state space is too big
        # We will create them lazily as we visit them.
        self.info_state_nodes: Dict[Tuple[int, str], InfoStateNode] = {}
        
        self.iteration = 0
    
    def _get_info_state_node(self, player: int, info_state: str, legal_actions: List[int]) -> InfoStateNode:
        """Lazily retrieves or creates an info state node."""
        key = (player, info_state)
        if key not in self.info_state_nodes:
            self.info_state_nodes[key] = InfoStateNode(legal_actions)
        return self.info_state_nodes[key]

    def _compute_counterfactual_regret(
        self,
        state,
        reach_probabilities: np.ndarray,
        player: int = None
    ) -> np.ndarray:
        """
        Recursive CFR traversal.
        """
        if state.is_terminal():
            returns = state.returns()
            if len(returns) > 0:
                return np.array(returns)
            return np.zeros(self.num_players)
        
        # Note: In Chance Sampling, we don't handle Chance Nodes recursively here.
        # We handled Chance at the root by sampling a specific state.
        
        # Decision node
        current_player = state.current_player()
        info_state = state.information_state_string(current_player)
        
        # Skip if probability of reaching this state is 0
        if np.all(reach_probabilities[:-1] == 0):
            return np.zeros(self.num_players)
        
        # LAZY INITIALIZATION: Get or create the node
        legal_actions = state.legal_actions()
        info_state_node = self._get_info_state_node(current_player, info_state, legal_actions)
        
        # Get current strategy (regret matching)
        current_strategy = regret_matching(
            info_state_node.cumulative_regret,
            info_state_node.legal_actions
        )
        
        # Compute state value and child utilities
        state_value = np.zeros(self.num_players)
        children_utilities = {}
        
        for action in legal_actions:
            action_prob = current_strategy.get(action, 0.0)
            new_state = state.child(action)
            
            # Update reach probabilities
            new_reach_probabilities = reach_probabilities.copy()
            new_reach_probabilities[current_player] *= action_prob
            
            # Recursive call
            child_utility = self._compute_counterfactual_regret(
                new_state, new_reach_probabilities, player
            )
            
            state_value += action_prob * child_utility
            children_utilities[action] = child_utility
        
        # Update Regrets and Policy
        # In Chance Sampling, we update everyone on the active path
        
        # 1. Counterfactual Reach Prob (prob of reaching here assuming player played to get here)
        # Product of all OTHER players' reach probs AND chance probability
        counterfactual_reach_prob = 1.0
        for p in range(self.num_players):
            if p != current_player:
                counterfactual_reach_prob *= reach_probabilities[p]
        # Include chance probability (important for chance sampling!)
        counterfactual_reach_prob *= reach_probabilities[-1]
        
        # 2. Update Regrets
        state_value_for_player = state_value[current_player]
        for action, action_prob in current_strategy.items():
            cfr_regret = counterfactual_reach_prob * (
                children_utilities[action][current_player] - state_value_for_player
            )
            info_state_node.cumulative_regret[action] += cfr_regret
            
            # 3. Update Average Policy (Linear Averaging)
            # Weighted by the reach prob of the player themselves
            info_state_node.cumulative_policy[action] += (
                self.iteration * reach_probabilities[current_player] * action_prob
            )
        
        return state_value
    
    def evaluate_and_update_policy(self):
        """Perform one iteration of Chance-Sampled CFR."""
        self.iteration += 1
        
        # 1. SAMPLE A NEW DEAL (The Fix)
        # This creates a fresh random shuffling of cards
        current_state = self.game.new_initial_state()
        
        # 2. Initialize reach probabilities (Chance prob is implicitly 1.0 for this sample)
        reach_probabilities = np.ones(self.num_players + 1)
        
        # 3. Run CFR on this specific deal
        self._compute_counterfactual_regret(
            current_state,
            reach_probabilities,
            player=None
        )
    
    def average_policy(self) -> Dict[Tuple[int, str], Dict[int, float]]:
        """Get average policy (normalized cumulative policy)."""
        avg_policy = {}
        for key, node in self.info_state_nodes.items():
            total = sum(node.cumulative_policy.values())
            if total > 0:
                avg_policy[key] = {
                    action: count / total
                    for action, count in node.cumulative_policy.items()
                }
            else:
                # Uniform if no counts
                uniform_prob = 1.0 / len(node.legal_actions)
                avg_policy[key] = {
                    action: uniform_prob
                    for action in node.legal_actions
                }
        return avg_policy
    
    def current_policy(self) -> Dict[Tuple[int, str], Dict[int, float]]:
        """Get current policy (from regret matching)."""
        policy = {}
        for key, node in self.info_state_nodes.items():
            policy[key] = regret_matching(node.cumulative_regret, node.legal_actions)
        return policy


def evaluate_policy(solver: CFRSolver, game: TinyCrewGame, num_episodes: int = 1000, 
                   use_average: bool = True, verbose: bool = True) -> Dict:
    """
    Evaluate the learned policy by simulating games.
    
    Args:
        solver: CFRSolver with learned policy
        game: TinyCrewGame instance
        num_episodes: Number of games to simulate
        use_average: If True, use average policy; if False, use current policy
        verbose: Print detailed statistics
    
    Returns:
        Dictionary with evaluation metrics
    """
    if use_average:
        policy = solver.average_policy()
        policy_name = "Average Policy"
    else:
        policy = solver.current_policy()
        policy_name = "Current Policy"
    
    total_rewards = []
    wins = 0
    total_tricks = []
    
    for episode in range(num_episodes):
        state = game.new_initial_state()
        tricks_played = 0
        
        while not state.is_terminal():
            if state.is_chance_node():
                # Sample chance outcome
                outcomes, probs = zip(*state.chance_outcomes())
                outcome = np.random.choice(outcomes, p=probs)
                state.apply_action(outcome)
            else:
                current_player = state.current_player()
                info_state = state.information_state_string(current_player)
                key = (current_player, info_state)
                
                # Get policy for this info state
                if key in policy:
                    action_probs = policy[key]
                    # Sample action according to policy
                    actions = list(action_probs.keys())
                    probs = list(action_probs.values())
                    action = np.random.choice(actions, p=probs)
                else:
                    # Uniform random if policy not found
                    legal_actions = state.legal_actions()
                    action = np.random.choice(legal_actions)
                
                state.apply_action(action)
                tricks_played += 1
        
        # Get final rewards
        returns = state.returns()
        if len(returns) > 0:
            reward = returns[0]  # All players get same reward in cooperative game
            total_rewards.append(reward)
            if reward >= 0.99:  # Win threshold (all tasks completed)
                wins += 1
            total_tricks.append(tricks_played)
    
    # Compute statistics
    total_rewards = np.array(total_rewards)
    win_rate = wins / num_episodes
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    avg_tricks = np.mean(total_tricks) if total_tricks else 0
    
    results = {
        'policy_name': policy_name,
        'num_episodes': num_episodes,
        'win_rate': win_rate,
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'min_reward': np.min(total_rewards),
        'max_reward': np.max(total_rewards),
        'avg_tricks': avg_tricks,
        'total_rewards': total_rewards
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Policy Evaluation: {policy_name}")
        print(f"{'='*60}")
        print(f"Episodes simulated: {num_episodes}")
        print(f"Win rate: {win_rate:.1%} ({wins}/{num_episodes})")
        print(f"Average reward: {avg_reward:.4f} ± {std_reward:.4f}")
        print(f"Reward range: [{np.min(total_rewards):.4f}, {np.max(total_rewards):.4f}]")
        print(f"Average tricks per game: {avg_tricks:.2f}")
        print(f"{'='*60}")
    
    return results


def compare_with_baseline(solver: CFRSolver, game: TinyCrewGame, num_episodes: int = 1000):
    """
    Compare learned policy with random baseline.
    
    Args:
        solver: CFRSolver with learned policy
        game: TinyCrewGame instance
        num_episodes: Number of games to simulate for each policy
    """
    print("\n" + "="*60)
    print("Policy Comparison: Learned vs Random Baseline")
    print("="*60)
    
    # Evaluate learned policy
    learned_results = evaluate_policy(solver, game, num_episodes, use_average=True, verbose=False)
    
    # Evaluate random baseline
    print("\nEvaluating random baseline...")
    random_rewards = []
    random_wins = 0
    
    for _ in range(num_episodes):
        state = game.new_initial_state()
        
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes, probs = zip(*state.chance_outcomes())
                outcome = np.random.choice(outcomes, p=probs)
                state.apply_action(outcome)
            else:
                legal_actions = state.legal_actions()
                action = np.random.choice(legal_actions)
                state.apply_action(action)
        
        returns = state.returns()
        if len(returns) > 0:
            reward = returns[0]
            random_rewards.append(reward)
            if reward >= 0.99:
                random_wins += 1
    
    random_rewards = np.array(random_rewards)
    random_win_rate = random_wins / num_episodes
    random_avg_reward = np.mean(random_rewards)
    
    # Print comparison
    print(f"\n{'Metric':<25} {'Learned Policy':<20} {'Random Baseline':<20} {'Improvement':<15}")
    print("-" * 80)
    print(f"{'Win Rate':<25} {learned_results['win_rate']:<20.1%} {random_win_rate:<20.1%} "
          f"{learned_results['win_rate'] - random_win_rate:>+14.1%}")
    print(f"{'Average Reward':<25} {learned_results['avg_reward']:<20.4f} {random_avg_reward:<20.4f} "
          f"{learned_results['avg_reward'] - random_avg_reward:>+14.4f}")
    
    improvement_pct = ((learned_results['win_rate'] / random_win_rate - 1) * 100) if random_win_rate > 0 else float('inf')
    print(f"\nRelative improvement: {improvement_pct:.1f}%")
    
    if learned_results['win_rate'] > random_win_rate + 0.05:  # 5% threshold
        print("✓ CFR learned significantly better strategies!")
    elif learned_results['win_rate'] > random_win_rate:
        print("~ CFR shows some improvement over random")
    else:
        print("⚠ CFR did not improve over random baseline")
    
    return {
        'learned': learned_results,
        'random': {
            'win_rate': random_win_rate,
            'avg_reward': random_avg_reward
        }
    }


def analyze_policy_quality(solver: CFRSolver, game: TinyCrewGame):
    """
    Analyze the quality of the learned policy by examining key statistics.
    
    Args:
        solver: CFRSolver with learned policy
        game: TinyCrewGame instance
    """
    print("\n" + "="*60)
    print("Policy Quality Analysis")
    print("="*60)
    
    avg_policy = solver.average_policy()
    
    # Statistics about the policy
    deterministic_states = 0
    uniform_states = 0
    total_states = len(avg_policy)
    
    for key, policy in avg_policy.items():
        probs = list(policy.values())
        max_prob = max(probs)
        min_prob = min(probs)
        
        # Check if policy is deterministic (one action has prob > 0.99)
        if max_prob > 0.99:
            deterministic_states += 1
        # Check if policy is uniform (all probs within 0.01 of each other)
        elif max_prob - min_prob < 0.01:
            uniform_states += 1
    
    print(f"\nPolicy Statistics:")
    print(f"  Total info states: {total_states}")
    print(f"  Deterministic states (>99% on one action): {deterministic_states} ({deterministic_states/total_states:.1%})")
    print(f"  Uniform states: {uniform_states} ({uniform_states/total_states:.1%})")
    print(f"  Mixed states: {total_states - deterministic_states - uniform_states} "
          f"({(total_states - deterministic_states - uniform_states)/total_states:.1%})")
    
    # Check regret convergence
    print(f"\nRegret Statistics:")
    all_regrets = []
    for key, node in solver.info_state_nodes.items():
        for regret in node.cumulative_regret.values():
            all_regrets.append(abs(regret))
    
    if all_regrets:
        print(f"  Mean absolute regret: {np.mean(all_regrets):.4f}")
        print(f"  Max absolute regret: {np.max(all_regrets):.4f}")
        print(f"  Regret convergence: {'Good' if np.mean(all_regrets) < 1.0 else 'Still learning'}")
    
    # Policy coverage
    print(f"\nPolicy Coverage:")
    print(f"  Info states with policy: {len(avg_policy)}")
    print(f"  Info states with regrets: {len(solver.info_state_nodes)}")

def main(num_iterations=5):
    """Main function for Chance-Sampled CFR."""
    print("="*60)
    print(f"Chance-Sampled CFR for Tiny Crew - {num_iterations} Iterations")
    print("="*60)
    
    game = TinyCrewGame()
    solver = CFRSolver(game)
    
    print("[1] Running iterations (sampling new deal every time)...")
    
    import time
    start_time = time.time()
    
    for i in range(num_iterations):
        solver.evaluate_and_update_policy()
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (num_iterations - i - 1)
            print(f"Iteration {i+1:5d}: "
                  f"InfoStates: {len(solver.info_state_nodes):5d} "
                  f"({elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining)")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Completed {num_iterations} iterations in {total_time:.1f}s")
    print(f"Average time per iteration: {total_time/num_iterations:.3f}s")
    print(f"Final Info States: {len(solver.info_state_nodes)}")
    
    # Print a sample strategy
    print("\n" + "-"*60)
    print("Sample Strategy (Player 0, first 3 info states):")
    print("-"*60)
    count = 0
    avg_policy = solver.average_policy()
    for key, node in solver.info_state_nodes.items():
        if key[0] == 0:  # Player 0
            player, info_state = key
            policy = avg_policy[key]
            print(f"\n  Info State: {info_state[:70]}...")
            print(f"  Policy: {dict(list(policy.items())[:5])}")
            print(f"  Regrets: {dict(list(node.cumulative_regret.items())[:5])}")
            count += 1
            if count >= 3:
                break
    
    print("\n" + "="*60)
    print(f"✓ {num_iterations} ITERATIONS COMPLETE!")
    print("="*60)
    
    # Evaluate the learned policy
    print("\n" + "="*60)
    print("EVALUATING LEARNED POLICY")
    print("="*60)
    
    # 1. Evaluate policy performance
    evaluate_policy(solver, game, num_episodes=1000, use_average=True, verbose=True)
    
    # 2. Compare with random baseline
    compare_with_baseline(solver, game, num_episodes=1000)
    
    # 3. Analyze policy quality
    analyze_policy_quality(solver, game)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    import sys
    num_iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    main(num_iterations=num_iterations)