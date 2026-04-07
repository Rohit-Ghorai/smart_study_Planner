#!/usr/bin/env python3
"""Test script for enhanced environment with more states and actions."""

from study_planner.config import StudyPlannerConfig
from study_planner.environment import StudyPlannerEnvironment
from study_planner.agent import QLearningAgent
from study_planner.trainer import train_agent, generate_greedy_schedule

def test_enhanced_environment():
    print("=" * 60)
    print("Testing Enhanced RL Study Planner Environment")
    print("=" * 60)
    
    # Create config
    config = StudyPlannerConfig.from_input(
        subjects=["Math", "Physics", "English"],
        difficulties=[4, 4, 2],
        strengths=[2, 3, 4],
        episodes=200,
    )
    
    print(f"\n✓ Config created: {len(config.subjects)} subjects")
    
    # Create environment
    env = StudyPlannerEnvironment(config)
    
    print(f"\n✓ Environment initialized")
    print(f"  - State space size: {env.num_states:,} states")
    print(f"    (slots={len(env.time_slots)} × energy=5 × action_type=5)")
    print(f"  - Action space size: {env.num_actions} actions")
    print(f"    • Base study: {len(env.base_actions)}")
    print(f"    • Deep study: {len(env.deep_study_actions)}")
    print(f"    • Revision: {len(env.revision_actions)}")
    print(f"    • Mock tests: {len(env.mock_test_actions)}")
    print(f"    • Breaks: {len(env.break_actions)}")
    
    # List all actions
    print(f"\n✓ Available actions:")
    for i, action in enumerate(env.actions, 1):
        print(f"  {i:2}. {action}")
    
    # Test reset and state
    state = env.reset()
    print(f"\n✓ Initial state: {state}")
    print(f"  - Time slot: {state[0]}")
    print(f"  - Energy level: {state[1]}/5")
    print(f"  - Last action type: {state[2]}")
    
    # Test a few steps
    print(f"\n✓ Simulating 3 steps:")
    for step in range(3):
        action_idx = step % len(env.actions)
        action_name = env.actions[action_idx]
        next_state, reward, done = env.step(action_idx)
        print(f"  Step {step+1}: {action_name:15} → reward={reward:6.2f} | next_state={next_state}")
        if done:
            print(f"  (Episode finished)")
            break
    
    # Test training
    print(f"\n✓ Training for 50 episodes...")
    agent = QLearningAgent(
        num_states=env.num_states,
        num_actions=env.num_actions,
        alpha=0.1,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        seed=42,
    )
    
    rewards = train_agent(env, agent, episodes=50, log_interval=0)
    
    total_reward = sum(rewards) / len(rewards)
    print(f"  Average reward over 50 episodes: {total_reward:.2f}")
    
    # Test greedy schedule generation
    print(f"\n✓ Generating greedy schedule...")
    schedule, schedule_reward = generate_greedy_schedule(env, agent)
    
    print(f"  Schedule generated: {len(schedule)} actions")
    print(f"  Total reward: {schedule_reward:.2f}")
    print(f"\n  Planned schedule:")
    for i, action in enumerate(schedule[:6], 1):
        print(f"    {i}. {action}")
    if len(schedule) > 6:
        print(f"    ... ({len(schedule) - 6} more actions)")
    
    print(f"\n{'='*60}")
    print(f"✅ All tests passed!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    test_enhanced_environment()
