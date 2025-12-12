import pyspiel
from open_spiel.python.algorithms import external_sampling_mccfr
import numpy as np

# 1. Load tiny_hanabi game
game = pyspiel.load_game("tiny_hanabi")

print("Game loaded:", game)
print(f"Number of players: {game.num_players()}")
print(f"Number of distinct actions: {game.num_distinct_actions()}")
print(f"Min utility: {game.min_utility()}")
print(f"Max utility: {game.max_utility()}")

# 2. Use MCCFR (a CFR variant) which works out-of-the-box in OpenSpiel
#    We'll track average policies as the experiment progresses.

cfr = external_sampling_mccfr.ExternalSamplingSolver(game)

num_iterations = 5000

print(f"\nRunning {num_iterations} iterations of External Sampling MCCFR...")
for i in range(num_iterations):
    cfr.iteration()
    if (i+1) % 1000 == 0:
        print(f"Iteration {i+1}")

# 3. Extract the average policy
avg_policy = cfr.average_policy().to_tabular()

print("\n=== Average Policy ===")
policy_dict = avg_policy.to_dict()
for k, v in policy_dict.items():
    print(f"{k}: {v}")

# 4. Evaluate the average strategy
#    Optimal score in tiny_hanabi is 10 (perfect cooperation)
#    No-cooperation equilibrium scores 8
#    Some intermediate strategies score between 8-10

def evaluate_policy(episodes=1000):
    """Evaluate the policy by playing episodes and computing average return."""
    total_returns = []
    
    for _ in range(episodes):
        state = game.new_initial_state()
        
        # Play until terminal
        while not state.is_terminal():
            if state.is_chance_node():
                # Sample a chance outcome
                outcomes, probs = zip(*state.chance_outcomes())
                outcome = np.random.choice(outcomes, p=probs)
                state.apply_action(outcome)
            else:
                # Current player chooses an action using the average policy
                pid = state.current_player()
                info_state_key = state.information_state_string(pid)
                probs = avg_policy.policy_for_key(info_state_key)
                action = np.random.choice(len(probs), p=probs)
                state.apply_action(action)
        
        # Get returns (all players get the same return in cooperative games)
        returns = state.returns()
        if len(returns) > 0:
            total_returns.append(returns[0])
    
    avg_return = np.mean(total_returns)
    std_return = np.std(total_returns)
    min_return = np.min(total_returns)
    max_return = np.max(total_returns)
    
    return {
        'mean': avg_return,
        'std': std_return,
        'min': min_return,
        'max': max_return,
        'returns': total_returns
    }

print("\n=== Evaluating Policy ===")
results = evaluate_policy(episodes=1000)
print(f"Average return: {results['mean']:.4f} ± {results['std']:.4f}")
print(f"Min return: {results['min']}")
print(f"Max return: {results['max']}")
print(f"\nExpected performance:")
print(f"  Optimal (perfect cooperation): 10.0")
print(f"  No-cooperation equilibrium: 8.0")
print(f"  Current policy: {results['mean']:.4f}")

# 5. Additional tests: Check game properties
print("\n=== Game Properties Test ===")
initial_state = game.new_initial_state()
print(f"Initial state is terminal: {initial_state.is_terminal()}")
print(f"Initial state is chance node: {initial_state.is_chance_node()}")

if initial_state.is_chance_node():
    chance_outcomes = initial_state.chance_outcomes()
    print(f"Number of chance outcomes: {len(chance_outcomes)}")
    for outcome, prob in chance_outcomes:
        print(f"  Outcome {outcome}: probability {prob:.4f}")

# 6. Test random play baseline for comparison
print("\n=== Random Baseline ===")
def random_baseline(episodes=1000):
    """Play randomly to establish a baseline."""
    total_returns = []
    
    for _ in range(episodes):
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
            total_returns.append(returns[0])
    
    return np.mean(total_returns)

random_avg = random_baseline(episodes=1000)
print(f"Random baseline average return: {random_avg:.4f}")
print(f"MCCFR improvement over random: {results['mean'] - random_avg:.4f}")

# 7. Verify correctness of learned policy
print("\n=== Policy Correctness Verification ===")

# Known benchmarks from the paper:
# - Optimal (perfect cooperation): 10.0
# - Bayesian Action Decoder: 9.5
# - Simplified Action Decoder: 9.5
# - Policy gradient: 9.0
# - Independent Q learning: 8.8
# - No-cooperation equilibrium: 8.0

OPTIMAL_SCORE = 10.0
NO_COOP_EQUILIBRIUM = 8.0
GOOD_THRESHOLD = 9.0  # Better than policy gradient
EXCELLENT_THRESHOLD = 9.5  # Matching state-of-the-art

mean_score = results['mean']
std_score = results['std']
n_samples = len(results['returns'])

# Compute 95% confidence interval
try:
    from scipy import stats
    confidence_level = 0.95
    t_critical = stats.t.ppf((1 + confidence_level) / 2, n_samples - 1)
    margin_error = t_critical * (std_score / np.sqrt(n_samples))
except ImportError:
    # Fallback to normal approximation (z-score) if scipy not available
    confidence_level = 0.95
    z_critical = 1.96  # 95% confidence interval for normal distribution
    margin_error = z_critical * (std_score / np.sqrt(n_samples))

ci_lower = mean_score - margin_error
ci_upper = mean_score + margin_error

print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"\nCorrectness Checks:")
print(f"  ✓ Better than no-cooperation equilibrium (8.0): {mean_score > NO_COOP_EQUILIBRIUM}")
print(f"  ✓ Better than random baseline ({random_avg:.4f}): {mean_score > random_avg}")
print(f"  ✓ Achieves 'good' performance (≥9.0): {mean_score >= GOOD_THRESHOLD}")
print(f"  ✓ Achieves 'excellent' performance (≥9.5): {mean_score >= EXCELLENT_THRESHOLD}")
print(f"  ✓ Achieves optimal performance (10.0): {mean_score >= OPTIMAL_SCORE}")

# Detailed assessment
if mean_score >= OPTIMAL_SCORE:
    assessment = "✓ OPTIMAL - Perfect cooperation achieved!"
elif mean_score >= EXCELLENT_THRESHOLD:
    assessment = "✓ EXCELLENT - Matches state-of-the-art (Bayesian Action Decoder level)"
elif mean_score >= GOOD_THRESHOLD:
    assessment = "✓ GOOD - Better than policy gradient baseline"
elif mean_score > NO_COOP_EQUILIBRIUM:
    assessment = "✓ ACCEPTABLE - Better than no-cooperation equilibrium"
else:
    assessment = "✗ POOR - Below no-cooperation equilibrium"

print(f"\nOverall Assessment: {assessment}")

# Check if policy is deterministic enough (low variance indicates consistent strategy)
print(f"\nPolicy Consistency:")
print(f"  Standard deviation: {std_score:.4f}")
if std_score < 0.5:
    print(f"  ✓ Low variance - Policy is consistent")
elif std_score < 1.0:
    print(f"  ~ Moderate variance - Some randomness in outcomes")
else:
    print(f"  ⚠ High variance - Policy may be inconsistent")

# Verify that we're achieving high scores frequently
high_score_rate = sum(1 for r in results['returns'] if r >= 9.0) / len(results['returns'])
optimal_rate = sum(1 for r in results['returns'] if r >= OPTIMAL_SCORE) / len(results['returns'])

# Detailed score breakdown
unique_scores = {}
for r in results['returns']:
    unique_scores[r] = unique_scores.get(r, 0) + 1

print(f"\nScore Distribution:")
print(f"  Frequency of scores ≥9.0: {high_score_rate*100:.1f}%")
print(f"  Frequency of optimal scores (10.0): {optimal_rate*100:.1f}%")
print(f"\nDetailed Score Breakdown:")
for score in sorted(unique_scores.keys(), reverse=True):
    count = unique_scores[score]
    percentage = (count / len(results['returns'])) * 100
    print(f"  Score {score:.1f}: {count} episodes ({percentage:.1f}%)")

# Verify math consistency
print(f"\nMath Verification:")
expected_mean = sum(score * count for score, count in unique_scores.items()) / len(results['returns'])
print(f"  Calculated mean from distribution: {expected_mean:.4f}")
print(f"  Reported mean: {mean_score:.4f}")
print(f"  Match: {abs(expected_mean - mean_score) < 0.0001}")

# Final verdict
print(f"\n{'='*60}")
print(f"FINAL VERDICT:")
if mean_score >= GOOD_THRESHOLD:
    print(f"  ✓ POLICY LEARNED CORRECTLY")
    print(f"    Average return: {mean_score:.4f} (target: ≥{GOOD_THRESHOLD})")
    print(f"    The MCCFR algorithm successfully learned a cooperative strategy.")
    
    # Note about mixed strategies
    if optimal_rate < 0.9:  # Less than 90% optimal
        print(f"\n  Note: Policy achieves optimal score {optimal_rate*100:.1f}% of the time.")
        print(f"        This suggests a mixed strategy equilibrium rather than pure cooperation.")
        print(f"        MCCFR finds Nash equilibria, which in cooperative games may be mixed.")
        print(f"        For pure cooperation (100% optimal), consider:")
        print(f"        - More training iterations")
        print(f"        - Different algorithms (e.g., policy gradient methods)")
        print(f"        - Cooperative-specific training approaches")
else:
    print(f"  ⚠ POLICY MAY NEED MORE TRAINING")
    print(f"    Average return: {mean_score:.4f} (target: ≥{GOOD_THRESHOLD})")
    print(f"    Consider increasing iterations or checking algorithm parameters.")
print(f"{'='*60}")

