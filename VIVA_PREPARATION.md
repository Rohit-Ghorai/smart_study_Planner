# Viva Preparation

1. **Why reinforcement learning?** It learns action preferences from feedback
   rather than requiring a fixed timetable.
2. **Why Q-learning?** It is off-policy and learns a greedy value estimate.
3. **Why SARSA?** It is on-policy and evaluates the exploratory policy.
4. **What is the state?** Time, energy, previous action, effort buckets, and
   break history.
5. **What is the action space?** Subject-specific regular, deep, revision,
   mock, and three break actions.
6. **Why cap progress?** To retain useful history without an unmanageable
   tabular state explosion.
7. **How is energy updated?** Study consumes bounded energy; breaks restore it.
8. **How is deep study represented?** It consumes two integer time slots.
9. **How are terminal states handled?** They receive no future-value term.
10. **What prevents mock-test reward hacking?** Its base value is bounded and
    repeated effort has diminishing returns.
11. **How are weak subjects prioritized?** Difficulty and confidence gap affect
    the need-based reward.
12. **What is epsilon?** The probability of exploratory action selection.
13. **Why decay epsilon?** Training shifts from exploration toward exploitation.
14. **How are experiments reproducible?** Random generators receive explicit
    seeds and both algorithms use independent tables.
15. **How is a schedule generated?** The trained policy is executed greedily
    from the reset state.
16. **How are weekly actions mapped?** Subject and activity suffixes are
    preserved, while break labels remain breaks.
17. **How are models secured?** Numeric archives are loaded with pickle disabled.
18. **What is validated on upload?** Size, dimensions, labels, finite values, and
    metadata.
19. **How is ownership enforced?** Plan queries include the authenticated user id.
20. **How are passwords stored?** Werkzeug password hashes, never plaintext.
21. **What does OpenEnv provide?** Independent task scoring and reset/step APIs.
22. **Why separate OpenEnv?** Its task MDP is an evaluation service, not the
    tabular study-planning trainer.
23. **What is adaptive replanning?** It recalculates remaining actions after a
    missed slot using current subject needs.
24. **What is the main limitation?** Compact buckets lose fine-grained history.
25. **What should be done before production?** Add CSRF protection, a managed
    session store, and multi-seed statistical experiments.

Further viva questions can be answered from the state, reward, persistence,
database, and deployment sections of the project report.
