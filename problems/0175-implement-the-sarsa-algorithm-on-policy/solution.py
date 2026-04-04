import numpy as np
import random

def sarsa_update(transitions, initial_states, alpha, gamma, max_steps):
    """
    Perform SARSA updates on the given environment transitions.
    Args:
        transitions (dict): mapping (state, action) -> (reward, next_state)
        initial_states (list): list of starting states to simulate episodes from
        alpha (float): learning rate
        gamma (float): discount factor
        max_steps (int): maximum steps allowed per episode
    Returns:
        dict: final Q-table as a dictionary {(state, action): value}
    """
    Q = {}
    state_to_actions = {}
    for s, a in transitions.keys():
        state_to_actions.setdefault(s, []).append(a)
        Q[(s, a)] = 0
    
    def greedy_action(state):
        return max(state_to_actions[state], key=lambda act: Q[state, act])

    for state in initial_states:
        if state not in state_to_actions.keys():
            continue
        act = greedy_action(state)
        for _ in range(max_steps):
            if (s, a) not in transitions:
                continue
            reward, next_state = transitions[(state, act)]
            if next_state == "terminal":
                Q[(state, act)] += alpha * (reward - Q[(state, act)])
                break
            else:
                next_act = greedy_action(next_state)
                Q[(state, act)] += alpha * (reward + gamma * Q[(next_state, next_act)] - Q[(state, act)])
            state, act = next_state, next_act
    return Q
    