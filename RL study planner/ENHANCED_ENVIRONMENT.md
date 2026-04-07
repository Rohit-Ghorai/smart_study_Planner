# Enhanced RL Study Planner Environment

## Overview
The study planner RL environment has been significantly expanded with **richer state representations** and **4x more actions**, creating a more realistic and sophisticated learning problem for the agent.

## State Space Expansion

### Previous State Representation
- **Simple tuple**: `(time_slot, last_subject)`
- **Size**: `3 slots × (3 subjects + 1 break + 1 None) = 16 states`

### New Enhanced State Representation
- **Tuple format**: `(time_slot, energy_level, last_action_type)`
- **State dimensions**:
  - `time_slot`: 0-2 (3 time periods: Morning, Afternoon, Evening)
  - `energy_level`: 1-5 (1=exhausted, 5=fresh)
  - `last_action_type`: "Idle", "Study", "Review", "Mock", "Break"
- **Total state space**: `3 × 5 × 5 = 75 states` (4.7x expansion)
- **Captures**: Energy management, study mode history, time awareness

## Action Space Expansion

### Previous Actions (4 total)
- Basic study per subject: `[Math, Physics, English, ...]`
- Break: `[Break]`

### New Enhanced Actions (3 × 5 = 15 total for 3 subjects)
1. **Base Study** (1 slot, regular effort)
   - `Math, Physics, English`
   - Reward: 5.0
   - Energy cost: -1 level

2. **Deep Study** (2 slots, intensive)
   - `Math_Deep, Physics_Deep, English_Deep`
   - Reward: 8.0 (highest for study sessions)
   - Energy cost: -1 level
   - Time cost: 2 slots

3. **Revision** (1 slot, light mode)
   - `Math_Rev, Physics_Rev, English_Rev`
   - Reward: 3.0 (lower, for reinforcement)
   - Energy cost: -1 level
   - Time cost: 1 slot

4. **Mock Tests** (1 slot, assessment)
   - `Math_Mock, Physics_Mock, English_Mock`
   - Reward: 10.0 (highest reward, knowledge evaluation)
   - Energy cost: -1 level
   - Time cost: 1 slot

5. **Breaks** (1 slot, recovery)
   - `Meditation`: Stress relief, +1.5 reward bonus
   - `Snack`: Nutrition, +0.5 reward bonus
   - `Water_Break`: Hydration, +0.0 reward bonus
   - Effect: +2 energy levels (max 5)

## Reward System Enhancements

### Per-Subject Rewards
```python
reward = base_reward + difficulty_bonus + adaptive_bonus + time_bonus + exam_bonus + diminishing + energy_penalty
```

**Components**:
- **Base reward**: 3-10 depending on action type
- **Difficulty bonus**: Scale 1-5 (higher difficulty = higher reward)
- **Adaptive bonus**: Student's gap between difficulty and confidence (-3 to +3)
- **Time pressure bonus**: Decreases as time runs out
- **Exam urgency bonus**: Days until exam (multiplier 0-3)
- **Diminishing returns**: Penalty if same action repeated too much (per subject)
- **Energy penalty**: Reduced rewards if energy too low (<3)

### Break Rewards
- **Quality break** (after study): +1.0 baseline
- **Meditation bonus**: +1.5 (stress management)
- **Snack bonus**: +0.5
- **Over-break penalty**: -3.0 (after 2+ breaks)
- **No-study penalty**: -2.0 (break before any study)

## Energy Management System

**Energy levels 1-5**:
- **Level 5**: Fresh, peak performance (start of day)
- **Level 3**: Normal, standard performance
- **Level 1**: Exhausted, energy penalty applied
- **Restoration**: Each break restores +2 levels (max 5)
- **Consumption**: Each study action costs -1 level

**Agent learns**: When to take breaks to maintain cognitive performance

## Subject Progress Tracking

Per-subject counters during each episode:
- `deep_study_count`: Tracks Deep study sessions
- `revision_count`: Tracks light revisions
- `mock_test_count`: Tracks practice exams
- `total_effort`: Cumulative effort metric

**Agent learns**: Balanced coverage across study types and subjects

## State-Action Dynamics

### What the Agent Learns
1. **Energy management**: When to take breaks given current energy
2. **Study mode selection**: Choose appropriate study intensity (base/deep/revision) for current state
3. **Assessment timing**: When to do mock tests (usually after adequate study)
4. **Subject prioritization**: Which subjects to focus on given time/exam urgency
5. **Break timing**: Different break types based on fatigue level
6. **Time efficiency**: Balance quality (Deep) with coverage (Base/Revision)

### State Transitions Example
```
Start: (0, 5, 'Idle')  # Time 0, Energy 5, No activity yet
  → [Action: Math_Deep]
→ (2, 4, 'Study')     # Time advances 2 slots, Energy drops to 4
  → [Action: Meditation]
→ (3, 5, 'Break')     # Time +1 slot, Energy restored to 5
  → [Action: Physics_Mock]
→ (4, 4, 'Study')     # Time +1 slot, Energy drops to 4 (done)
```

## Training Efficiency

**Larger state space** enables:
- More nuanced decision-making
- Energy-aware scheduling
- Natural work-rest cycles emerge from training
- Better generalization to real study scenarios

**Larger action space** enables:
- Sampling various study strategies
- Q-Learning discovers which actions are best at which times
- Emergent complex behaviors (e.g., Mock tests as confidence checks)

## Statistics

| Metric | Value |
|--------|-------|
| State space size | 75 (vs 16 before; 4.7x) |
| Action space size | 15 (vs 4 before; 3.75x) |
| Q-table parameters | 1,125 (=75×15) |
| Reward components | 7 per study action |
| Energy levels | 5 discrete levels |
| Subjects tracked | Dynamic (N per config) |

## Test Results

```
Average reward over 50 episodes: 26.01
Generated schedule: 2 actions
Final reward: 31.50
Schedule discovered: [Math_Mock, Math_Deep]
(Agent learns high-value actions like mock tests and deep study)
```

## Integration

- **Backward compatible**: Old code paths still work
- **Web integration**: New actions displayed in UI templates
- **Training**: Supports both Q-Learning and SARSA with new states/actions
- **Explanations**: Explainable AI updates for new action types
