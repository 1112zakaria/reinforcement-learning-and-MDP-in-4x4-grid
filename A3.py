#!/usr/bin/env python3
####################################################
# COMP3106 - Introduction to Artificial Intelligence
# Assignment 3
#
# April 2022, Zakaria Ismail - 101143497
#
# Copyright (c) 2022 by Zakaria Ismail
# All rights reserved.
####################################################

from enum import Enum

MAX_I = 4
MAX_J = 4


class Action(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    EMPTY = 4


def _get_resulting_state(state, action):
    """
    S(state, action) = state'
    """
    i = state[0]
    j = state[1]

    if action is Action.LEFT:
        return [i, j-1] if j-1 >= 0 else [i, j]

    if action is Action.RIGHT:
        return [i, j+1] if j+1 < MAX_J else [i, j]

    if action is Action.UP:
        return [i-1, j] if i-1 >= 0 else [i, j]

    if action is Action.DOWN:
        return [i+1, j] if i+1 < MAX_I else [i, j]

    return None


def _get_neighbour_states(state):
    """
    N(state) = set of neighbouring states
    """
    neighbour_states = []
    i = state[0]
    j = state[1]

    if i-1 >= 0:
        neighbour_states += [[i-1, j]]
    if i+1 < MAX_I:
        neighbour_states += [[i+1, j]]
    if j-1 >= 0:
        neighbour_states += [[i, j-1]]
    if j+1 < MAX_J:
        neighbour_states += [[i, j+1]]

    return neighbour_states


def _get_triggered_actions(action):
    """
    G(action) = set of two other actions that can be triggered
    """
    transitions = {
        Action.LEFT: [Action.UP, Action.DOWN],
        Action.RIGHT: [Action.UP, Action.DOWN],
        Action.UP: [Action.LEFT, Action.RIGHT],
        Action.DOWN: [Action.LEFT, Action.RIGHT]
    }
    return transitions[action]


def _get_resulting_two_actions_states(state, transition_states):
    """
    M(s,G(a)) = set of resulting states from s when
    when applying the two actions triggered by action
    a
    """
    resulting_states = []
    for transition_state in transition_states:
        resulting_states += [_get_resulting_state(state, transition_state)]
    return resulting_states


def _get_transition_model(noiseP, state, action, next_state):
    """
    P(s,a,s') = stochastic transition model
    """
    S_sa = _get_resulting_state(state, action)
    M_sGa = _get_resulting_two_actions_states(
        state, _get_triggered_actions(action))
    if next_state == S_sa and next_state not in M_sGa:
        return 1 - noiseP
    if next_state == S_sa and next_state in M_sGa:
        return 1 - noiseP/2
    if next_state != S_sa and next_state in M_sGa:
        return noiseP/2
    if next_state != S_sa and next_state not in M_sGa:
        return 0

    return None


def _get_max_expected_util(U, noiseP, state):
    """
    max action with sum for all s' P(s'|s,a)*U(s') expected util
    and resulting state
    """
    max_exp_util = None
    max_state = None
    for action in [Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN]:
        # Calculate exp util
        exp_util = 0
        resulting_state = _get_resulting_state(state, action)
        triggered_actions = _get_triggered_actions(action)
        action_state_1 = _get_resulting_state(state, triggered_actions[0])
        action_state_2 = _get_resulting_state(state, triggered_actions[1])

        next_states = [resulting_state, action_state_1, action_state_2]
        actions = [action, triggered_actions[0], triggered_actions[1]]
        #resulting_two_actions_states = _get_resulting_two_actions_states(
        #    state, _get_triggered_actions(action))
        #next_states = [resulting_state] + [resulting_two_actions_states]
        #import pdb; pdb.set_trace()
        for i in range(len(next_states)):
            next_state = next_states[i]
            action = actions[i]
            trans_model = _get_transition_model(noiseP, state, action, next_state)
            #print(f"Next state: {next_state}")
            exp_util += trans_model * U[next_state[0]][next_state[1]]
        
        if max_exp_util is None or exp_util > max_exp_util:
            max_exp_util = exp_util
            max_state = resulting_state

    return max_exp_util, max_state

def _get_reward(vState, fState, victoryR, failureR, state):
    if state == vState:
        return victoryR
    if state == fState:
        return failureR
    return 0

def value_iteration(noiseP, vState, fState, victoryR, failureR, gamma, max_iternum):
    """
    Pseudocode:

    def VALUE-ITERATION(mdp, epsilon):
        INPUTS:
            - mdp, MDP w/ states S, actions A(s), transition model
                P(s'|s,a), rewards R(s), discount gamma
            - epsilon, maximum error allowed in the utility of any
                state in an iteration
        RETURNS a utility function
        LOCAL VARIABLES:
            - U, U', vectors of utilities for states in S, initially 0
            - sigma, maximum change in the utility of any state in an
                iteration

        do:
            U = U'
            sigma = 0
            for s in states S:
                U'[s] = R(s) + gamma * max( sum( P(s'|s,a)*U[s'] ) )

                if |U'[s] - U[s]| > sigma:
                    sigma = |U'[s] - U[s]|
        until sigma < epsilon(1-gamma)/gamma
        return U

    - OBSERVATION: instead of using error and maximum error change in the
        utility of any state to determine whether to continue iterating,
        we are using an input number of iterations to set the number of
        times we iterate
    """
    # Create MDP obj (S, A(s), P(s'|s,a), R(s), )
    U = [[0 for i in range(MAX_I)] for j in range(
        MAX_J)]   # states, i=row=y, j=column=y
    # Set reward and fail state
    U[vState[0]][vState[1]] = victoryR
    U[fState[0]][fState[1]] = failureR

    # Iterate max_iternum times
    for i in range(max_iternum - 1):
        # Iterate over each neighboring state, that is not reward/fail state
        # Idea: use queue
        #state_queue = []
        # OR
        # Iterate over each state, that is not reward/fail state
        for i in range(MAX_I):
            for j in range(MAX_J):
                # break if vState or fState
                if [i, j] == [vState] or [i, j] == [fState]:
                    break
                state = [i,j]
                # for each action, get the expected util of all the
                #   resulting states of the transition states
                # note:
                #   - if transition state not in neighbour
                max_exp_util, next_state = _get_max_expected_util(U, noiseP, state)
                reward = _get_reward(vState, fState, victoryR, failureR, next_state)
                U[i][j] = reward + gamma * max_exp_util

    return U


def QL_explore(noiseP, initS0, vState, fState, victoryR, failureR, gamma, alpha, epsilon, max_iternum):
    pass


if __name__ == "__main__":
    print(value_iteration(0.3,    [3, 3],    [2, 2],    20,    -10,    0.9,    500))
