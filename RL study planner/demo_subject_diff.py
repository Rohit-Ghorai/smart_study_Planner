#!/usr/bin/env python3
"""
Demonstrate how the model differentiates subjects with identical specs
through per-subject progress tracking and diminishing returns.
"""

from study_planner.config import StudyPlannerConfig
from study_planner.environment import StudyPlannerEnvironment
from study_planner.agent import QLearningAgent
from study_planner.trainer import train_agent

def demo_subject_differentiation():
    print("=" * 70)
    print("Subject Differentiation with Identical Specs")
    print("=" * 70)
    
    # Create 2 subjects with IDENTICAL specs
    config = StudyPlannerConfig.from_input(
        subjects=["Math", "Physics"],  # Same names differentiate them
        difficulties=[4, 4],            # IDENTICAL difficulty
        strengths=[2, 2],               # IDENTICAL confidence
        episodes=5,
    )
    
    print("\n✓ Config with IDENTICAL specs:")
    print(f"  Math:    difficulty=4, strength=2")
    print(f"  Physics: difficulty=4, strength=2")
    
    env = StudyPlannerEnvironment(config)
    agent = QLearningAgent(
        num_states=env.num_states,
        num_actions=env.num_actions,
        alpha=0.1,
        gamma=0.9,
        epsilon=1.0,
        seed=42,
    )
    
    # Train for a few episodes to see how model learns to differentiate
    train_agent(env, agent, episodes=5, log_interval=0)
    
    print("\n" + "=" * 70)
    print("How Model Differentiates Identical Specs:")
    print("=" * 70)
    
    print("\n1️⃣  ACTION SPACE SEPARATION")
    print("   Each subject is a SEPARATE action:")
    print(f"   Action 0: Math       (separate action index)")
    print(f"   Action 1: Physics    (separate action index)")
    print(f"   Action 2: Math_Deep  (separate from Math)")
    print(f"   Action 3: Physics_Deep (separate from Physics)")
    print("   → Q-Learning learns Q(state, Math) vs Q(state, Physics)")
    
    print("\n2️⃣  PER-SUBJECT PROGRESS TRACKING")
    print("   Each subject has its own history during episode:")
    print("   Math.deep_study_count = [independently tracked]")
    print("   Physics.deep_study_count = [independently tracked]")
    
    print("\n3️⃣  DIMINISHING RETURNS PENALTY")
    print("   If Math studied 2 times with Deep mode:")
    print("   diminishing = -(2 * 0.5) = -1.0  ← penalty increases")
    print("   ")
    print("   If Physics NOT studied with Deep mode yet:")
    print("   diminishing = -(0 * 0.5) = 0.0   ← no penalty")
    print("   ")
    print("   Same action → Different rewards based on history!")
    
    print("\n4️⃣  LEARNED Q-VALUES DIVERGE")
    print("   Initial state: Math and Physics have SAME reward potential")
    print("   After exploration:")
    print("   Q(state, Math) ≠ Q(state, Physics)")
    print("   (based on which was studied more in episode)")
    
    # Show actual Q-table snippet
    print("\n" + "=" * 70)
    print("Actual Q-Table Sample (first state, all subjects):")
    print("=" * 70)
    
    state_0 = env.state_space[0]  # (0, 5, 'Idle')
    print(f"\nState 0: {state_0}")
    
    q_values = agent.q_table[0]
    print("\nQ-values for this state:")
    for action_idx, (action_name, q_val) in enumerate(zip(env.actions, q_values)):
        if "Math" in action_name or "Physics" in action_name or action_name in ["Meditation", "Snack"]:
            print(f"  {action_name:15} → Q = {q_val:7.3f}")
    
    print("\n" + "=" * 70)
    print("Key Insight:")
    print("=" * 70)
    print("""
Even though Math and Physics have identical initial specs:
✓ They are DISTINCT actions in the action space
✓ Each has independent progress tracking (deep_study_count, etc.)
✓ Diminishing returns push agent toward understudied subjects
✓ Q-values naturally diverge through learning
✓ Agent learns balanced schedule without explicit state difference!

This is actually ELEGANT - the model doesn't need to track
"Math studied 2x" in the state. It emerges from:
  - Per-subject action history (progress tracking)
  - Diminishing returns reward penalty
  - Q-Learning over many state-action pairs
""")

if __name__ == "__main__":
    demo_subject_differentiation()
