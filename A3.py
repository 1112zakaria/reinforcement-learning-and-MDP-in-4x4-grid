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
from pprint import pprint
import random

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


def _get_transition_model(noiseP, S_sa, M_sGa, next_state):
    """
    P(s,a,s') = stochastic transition model
    """
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
        two_action_states = [action_state_1, action_state_2]

        next_states = [resulting_state, action_state_1, action_state_2]
        # resulting_two_actions_states = _get_resulting_two_actions_states(
        #    state, _get_triggered_actions(action))
        #next_states = [resulting_state] + [resulting_two_actions_states]
        #import pdb; pdb.set_trace()
        for next_state in next_states:
            trans_model = _get_transition_model(
                noiseP, resulting_state, two_action_states, next_state)
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


def _get_max_expected_util_v2(U, noiseP, vState, fState, victoryR, failureR, gamma, state):
    max_exp_util = None
    for action in [Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN]:
        #print(action)
        exp_util = 0
        resulting_state = _get_resulting_state(state, action)
        triggered_actions = _get_triggered_actions(action)
        action_state_1 = _get_resulting_state(state, triggered_actions[0])
        action_state_2 = _get_resulting_state(state, triggered_actions[1])
        two_action_states = [action_state_1, action_state_2]
        #next_states = [resulting_state, action_state_1, action_state_2]
        next_states = []
        for state in [resulting_state, action_state_1, action_state_2]:
            if state not in next_states:
                next_states += [state]

        for next_state in next_states:
            trans_model = _get_transition_model(
                noiseP, resulting_state, two_action_states, next_state)
            reward = _get_reward(vState, fState, victoryR,
                                 failureR, next_state)
            #print(f"trans model: {trans_model}")
            #exp_util += trans_model * \
            #    (reward + gamma * U[next_state[0]][next_state[1]])   # removed reward
            #exp_util += trans_model * U[next_state[0]][next_state[1]]
            local_util = (trans_model * (reward + gamma * U[next_state[0]][next_state[1]]))
            #print(f"({trans_model} * ({reward} + {gamma} * {U[next_state[0]][next_state[1]]})) = {local_util}")
            exp_util += local_util

        #print()
        #print(exp_util)
        if max_exp_util is None or exp_util > max_exp_util:
            max_exp_util = exp_util

    return max_exp_util


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
    U_prime = [[0 for i in range(MAX_I)] for j in range(
        MAX_J)]   # states, i=row=y, j=column=y
    # Set reward and fail state
    #U_prime[vState[0]][vState[1]] = victoryR   # changed from victoryR
    #U_prime[fState[0]][fState[1]] = failureR  # changed from failureR

    # Iterate max_iternum times
    for i in range(max_iternum):
        # Iterate over each neighboring state, that is not reward/fail state
        # Idea: use queue
        #state_queue = []
        # OR
        # Iterate over each state, that is not reward/fail state
        #print(U_prime)
        #import pdb;pdb.set_trace()
        U = U_prime.copy()
        for i in range(MAX_I):
            for j in range(MAX_J):
                # break if vState or fState
                state = [i, j]
                if state == vState or state == fState:
                    break
                # for each action, get the expected util of all the
                #   resulting states of the transition states
                # note:
                #   - if transition state not in neighbour

                # Old code
                # max_exp_util, next_state = _get_max_expected_util(
                #     U, noiseP, state)
                # reward = _get_reward(
                #     vState, fState, victoryR, failureR, next_state)
                # U_prime[i][j] = reward + gamma * max_exp_util

                # New code
                #print(f"State: {state}")
                U_prime[i][j] = _get_max_expected_util_v2(
                    U, noiseP, vState, fState, victoryR, failureR, gamma, state)

    return U_prime


def QL_explore(noiseP, initSO, vState, fState, victoryR, failureR, gamma, alpha, epsilon, max_iternum):
    """
    Pseudocode:

    def EPSILON-GREEDY-Q-LEARNING:
        INPUTS: 
            - alpha, learning rate
            - gamma, discount factor
            - epsilon, small number
        RESULT:
            - Q-table containing Q(S,A) pairs defining
                estimated optimal policy pi_star
        
        Initialize Q(s,a) except Q(terminal)
        Q(terminal) <- 0

        for each episode do
            Initialize state S
            for each step in episode do
                do
                    A <- SELECT-ACTION(Q,S,epsilon)
                    Q(S,A) <- Q(S,A) + alpha * (R + gamma max-A (Q(S',a) - Q(S,A)) )
                    S <- S'
                while S is not terminal
            
    """

    # Initialize every Q-table value
    policy = [[Action.RIGHT for j in range(MAX_J)] for i in range(MAX_I)]
    Q_table = [[[Action.RIGHT.value for k in range(4)] for j in range(MAX_J)] for i in range(MAX_I)]
    U = [[1 for j in range(MAX_J)] for i in range(MAX_I)]
    
    policy[vState[0]][vState[1]] = None
    policy[fState[0]][fState[1]] = None
    Q_table[vState[0]][vState[1]] = [0,0,0,0]
    Q_table[fState[0]][fState[1]] = [0,0,0,0]

    # What is an episode? Assume iteration.
    # Is number of iterations number of steps total? idk
    for episode_num in range(max_iternum):
        S = initSO  # initialize state S
        num_steps = 0
        while (S != fState or S != vState) and num_steps < max_iternum:
            A = select_action(Q_table, S, epsilon)
            S_prime = _get_resulting_state(S, A)
            # Take action A, then observe reward R and next state S'
            
            Q_sa = Q_table[S[0]][S[1]][A.value]
            max_util = Q_table[S_prime[0]][S_prime[1]][A.value] - Q_sa
            #actions = _get_triggered_actions(A)
            for action in [Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN]:
                util = Q_table[S_prime[0]][S_prime[1]][action.value] - Q_sa
                if util > max_util:
                    max_util = util

            reward = _get_reward(vState, fState, victoryR, failureR, S_prime)

            Q_table[S[0]][S[1]][A.value] = Q_table[S[0]][S[1]][A.value] + alpha * (reward + gamma * max_util)
            S = S_prime
            num_steps += 1
    
    return Q_table

def select_action(Q, S, epsilon):
    """
    Pseudocode:
    
    def SELECT-ACTION(Q, S, epsilon):
        DATA:
            - Q: Q-table generated so far
            - S: current state
            - epsilon: a small number?
        RESULT: selected action

        n <- uniform random number between 0 and 1
        if n < epsilon:
            A <- random action from the action space
        else:
            A <- maxQ(S)
        
        return A
    """
    n = random.uniform(0,1)
    A = None
    if n < epsilon:
        # Select random action from action space
        A = Action(random.randint(0,3))
        pass
    else:
        # Select action with max Q(S,A)
        A = Action.LEFT
        max_Q = Q[S[0]][S[1]][A.value]
        for i in range(4):
            if Q[S[0]][S[1]][i] > max_Q:
                A = Action(i)

    return A

if __name__ == "__main__":
    #pprint(value_iteration(0.3, [3, 3], [2, 2], 20, -10, 0.9, 500))
    pprint(value_iteration(0.2, [0,3], [1,3], 1, -1, 0.8, 10))
    #pprint(QL_explore(0.0, [0, 0], [3, 3], [3, 2], 20, -10, 0.9, 0.3, 0.3, 500))
