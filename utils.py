import numpy as np

from typing import List
from copy import deepcopy
from layout import Layout
from greedy import greedy_solve


def _compactState(yard):
    sort = []
    for stack in yard:
      for container in stack:
        if not container in sort:
          sort.append(container)
    sort = sorted(sort)
    maxValue = 0
    for i in range(len(yard)):
      for j in range(len(yard[i])):
        yard[i][j] = sort.index(yard[i][j]) + 1
        if yard[i][j] > maxValue:
          maxValue = yard[i][j]
    return yard

def _elevateState(yard, h, max_item):
    for stack in yard:
      while len(stack) < h:
        stack.insert(0,1.2*max_item)
    return yard

def _flattenState(state):
    flatten = []
    for lista in state:
        for item in lista:
            flatten.append(item)
    return flatten

def _normalize(state,max_item):
    array = np.array(state)
    return array / max_item

def prepare(state, height):
    max_item = max(set().union(*state))

    state = _compactState(state)
    state = _elevateState(state, height, max_item)
    state = _normalize(state, max_item)
    state = _flattenState(state)

    return state


def _compute_unsorted_elements(stacks):
    unsorted_elements = []

    for stack in stacks:
        if len(stack) == 0: 
            unsorted_elements.append(0)
            continue
        
        sorted_elements = 1
        while(sorted_elements<len(stack) and stack[sorted_elements] <= stack[sorted_elements-1]):
            sorted_elements +=1
        
        unsorted_elements.append(len(stack) - sorted_elements)
    
    return unsorted_elements

def process_state(state: List[int]):
    state = deepcopy(state)
    unsorted_elements = _compute_unsorted_elements(state)
    percentages = [elem / len(unsorted_elements) for elem in unsorted_elements]
    state = prepare(state, 7)
    
    for percentage in percentages:
        state.append(percentage)

    return np.array(state)