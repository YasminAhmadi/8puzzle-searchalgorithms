from collections import deque
import heapq
import time
import tracemalloc
import signal
import time

class State:
    def __init__(self, puzzle, g_val, parent):
        self.puzzle = puzzle
        self.g_val = g_val
        self.parent = parent
        
    def __eq__(self, other):
        return self.puzzle == other.puzzle

    def __lt__(self, other):
        return self.g_val < other.g_val

def get_children(state, goal_state, g_val):
    zero_index = state.index(0)
    children = []

    # Up
    if zero_index not in range(0, 3):
        new_puzzle = state[:]
        new_puzzle[zero_index], new_puzzle[zero_index-3] = new_puzzle[zero_index-3], new_puzzle[zero_index]
        children.append(State(new_puzzle, g_val + 1, state))

    # Down
    if zero_index not in range(6, 9):
        new_puzzle = state[:]
        new_puzzle[zero_index], new_puzzle[zero_index+3] = new_puzzle[zero_index+3], new_puzzle[zero_index]
        children.append(State(new_puzzle, g_val + 1, state))

    # Left
    if zero_index not in [0, 3, 6]:
        new_puzzle = state[:]
        new_puzzle[zero_index], new_puzzle[zero_index-1] = new_puzzle[zero_index-1], new_puzzle[zero_index]
        children.append(State(new_puzzle, g_val + 1, state))

    # Right
    if zero_index not in [2, 5, 8]:
        new_puzzle = state[:]
        new_puzzle[zero_index], new_puzzle[zero_index+1] = new_puzzle[zero_index+1], new_puzzle[zero_index]
        children.append(State(new_puzzle, g_val + 1, state))

    return children

"""# BFS"""

def bfs_8puzzle(start, goal):
    visited = set()
    queue = deque([(start, [])])
    while queue:
        state, actions = queue.popleft()
        if state == goal:
            return actions, len(actions)
        if str(state) in visited:
            continue
        visited.add(str(state))
        zero_index = state.index(0)
        children = []
        if zero_index not in range(0, 3):
            new_state = state[:]
            new_state[zero_index], new_state[zero_index - 3] = new_state[zero_index - 3], new_state[zero_index]
            children.append(("up", new_state))
        if zero_index not in range(6, 9):
            new_state = state[:]
            new_state[zero_index], new_state[zero_index + 3] = new_state[zero_index + 3], new_state[zero_index]
            children.append(("down", new_state))
        if zero_index not in [0, 3, 6]:
            new_state = state[:]
            new_state[zero_index], new_state[zero_index - 1] = new_state[zero_index - 1], new_state[zero_index]
            children.append(("left", new_state))
        if zero_index not in [2, 5, 8]:
            new_state = state[:]
            new_state[zero_index], new_state[zero_index + 1] = new_state[zero_index + 1], new_state[zero_index]
            children.append(("right", new_state))
        for direction, child in children:
            queue.append((child, actions + [direction]))
    return [], 0

"""# DFS"""

def dfs_8puzzle(start, goal):
    visited = set()
    stack = [(start, [])]
    while stack:
        state, actions = stack.pop()
        if state == goal:
            return actions, len(actions)
        if str(state) in visited:
            continue
        visited.add(str(state))
        zero_index = state.index(0)
        children = []
        if zero_index not in range(0, 3):
            new_state = state[:]
            new_state[zero_index], new_state[zero_index - 3] = new_state[zero_index - 3], new_state[zero_index]
            children.append(("up", new_state))
        if zero_index not in range(6, 9):
            new_state = state[:]
            new_state[zero_index], new_state[zero_index + 3] = new_state[zero_index + 3], new_state[zero_index]
            children.append(("down", new_state))
        if zero_index not in [0, 3, 6]:
            new_state = state[:]
            new_state[zero_index], new_state[zero_index - 1] = new_state[zero_index - 1], new_state[zero_index]
            children.append(("left", new_state))
        if zero_index not in [2, 5, 8]:
            new_state = state[:]
            new_state[zero_index], new_state[zero_index + 1] = new_state[zero_index + 1], new_state[zero_index]
            children.append(("right", new_state))
        for direction, child in children:
            stack.append((child, actions + [direction]))
    return [], 0

"""# IDS"""

def ids_8puzzle(start, goal):
    def dfs_limited(state, goal, limit, path):
        if state == goal:
            return path
        if limit == 0:
            return None
        zero_index = state.index(0)
        children = []
        if zero_index not in range(0, 3):
            new_state = state[:]
            new_state[zero_index], new_state[zero_index - 3] = new_state[zero_index - 3], new_state[zero_index]
            children.append(("up", new_state))
        if zero_index not in range(6, 9):
            new_state = state[:]
            new_state[zero_index], new_state[zero_index + 3] = new_state[zero_index + 3], new_state[zero_index]
            children.append(("down", new_state))
        if zero_index not in [0, 3, 6]:
            new_state = state[:]
            new_state[zero_index], new_state[zero_index - 1] = new_state[zero_index - 1], new_state[zero_index]
            children.append(("left", new_state))
        if zero_index not in [2, 5, 8]:
            new_state = state[:]
            new_state[zero_index], new_state[zero_index + 1] = new_state[zero_index + 1], new_state[zero_index]
            children.append(("right", new_state))
        for direction, child in children:
            result = dfs_limited(child, goal, limit - 1, path + [direction])
            if result is not None:
                return result
        return None

    for depth in range(0, 100):
        result = dfs_limited(start, goal, depth, [])
        if result is not None:
            return result, len(result)
    return [], 0

"""# UCS"""

import heapq



def ucs_8puzzle(start, goal):
    visited = set()
    heap = [(0, (start, []))]
    while heap:
        g_val, (state, actions) = heapq.heappop(heap)
        if state == goal:
            return actions, g_val
        if str(state) in visited:
            continue
        visited.add(str(state))
        zero_index = state.index(0)
        children = []
        if zero_index not in range(0, 3):
            new_state = state[:]
            new_state[zero_index], new_state[zero_index - 3] = new_state[zero_index - 3], new_state[zero_index]
            children.append(("up", new_state))
        if zero_index not in range(6, 9):
            new_state = state[:]
            new_state[zero_index], new_state[zero_index + 3] = new_state[zero_index + 3], new_state[zero_index]
            children.append(("down", new_state))
        if zero_index not in [0, 3, 6]:
            new_state = state[:]
            new_state[zero_index], new_state[zero_index - 1] = new_state[zero_index - 1], new_state[zero_index]
            children.append(("left", new_state))
        if zero_index not in [2, 5, 8]:
            new_state = state[:]
            new_state[zero_index], new_state[zero_index + 1] = new_state[zero_index + 1], new_state[zero_index]
            children.append(("right", new_state))
        for direction, child in children:
            heapq.heappush(heap, (g_val + 1, (child, actions + [direction])))
    return [], 0

"""# A*"""

def heuristic(puzzle, goal_state):
    h_val = 0
    for i in range(9):
        if puzzle[i] == 0:
            continue
        h_val += abs(i // 3 - goal_state.index(puzzle[i]) // 3) + abs(i % 3 - goal_state.index(puzzle[i]) % 3)
    return h_val

def astar_8puzzle(start, goal_state):
    visited = set()
    heap = [State(start, 0, None)]
    while heap:
        state = heapq.heappop(heap)
        if state.puzzle == goal_state:
            actions = []
            while state.parent is not None:
                zero_index = state.puzzle.index(0)
                parent_index = state.parent.puzzle.index(0)
                if parent_index - zero_index == 3:
                    actions.append("up")
                elif parent_index - zero_index == -3:
                    actions.append("down")
                elif parent_index - zero_index == 1:
                    actions.append("left")
                else:
                    actions.append("right")
                state = state.parent
            actions.reverse()
            return actions, len(actions)
        if str(state.puzzle) in visited:
            continue
        visited.add(str(state.puzzle))
        children = get_children(state.puzzle, goal_state, state.g_val)
        for child in children:
            f_val = child.g_val + heuristic(child.puzzle, goal_state)
            heapq.heappush(heap, State(child.puzzle, f_val, state))
    return [], 0

"""# Testing"""

# bfs_8puzzle
# dfs_8puzzle
# ids_8puzzle
# ucs_8puzzle
# astar_8puzzle

start_time = time.perf_counter()
tracemalloc.start()
print(bfs_8puzzle([0,1,3,4,2,5,7,8,6], [1,2,3,4,5,6,7,8,0]))
end_time = time.perf_counter()
_, memory_usage = tracemalloc.get_traced_memory()
print(f"Time taken BFS: {end_time - start_time:.6f} seconds")
print(f"Memory usage BFS: {memory_usage / (1024 * 1024):.2f} MB")
print()
start_time = time.perf_counter()
tracemalloc.start()
print(dfs_8puzzle([0,1,3,4,2,5,7,8,6], [1,2,3,4,5,6,7,8,0]))
end_time = time.perf_counter()
_, memory_usage = tracemalloc.get_traced_memory()
print(f"Time taken DFS: {end_time - start_time:.6f} seconds")
print(f"Memory usage DFS: {memory_usage / (1024 * 1024):.2f} MB")
print()
start_time = time.perf_counter()
tracemalloc.start()
print(ids_8puzzle([0,1,3,4,2,5,7,8,6], [1,2,3,4,5,6,7,8,0]))
end_time = time.perf_counter()
_, memory_usage = tracemalloc.get_traced_memory()
print(f"Time taken IDS: {end_time - start_time:.6f} seconds")
print(f"Memory usage IDS: {memory_usage / (1024 * 1024):.2f} MB")
print()
start_time = time.perf_counter()
tracemalloc.start()
print(ucs_8puzzle([0,1,3,4,2,5,7,8,6], [1,2,3,4,5,6,7,8,0]))
end_time = time.perf_counter()
_, memory_usage = tracemalloc.get_traced_memory()
print(f"Time taken UCS: {end_time - start_time:.6f} seconds")
print(f"Memory usage UCS: {memory_usage / (1024 * 1024):.2f} MB")
print()
start_time = time.perf_counter()
tracemalloc.start()
print(astar_8puzzle([0,1,3,4,2,5,7,8,6], [1,2,3,4,5,6,7,8,0]))
end_time = time.perf_counter()
_, memory_usage = tracemalloc.get_traced_memory()
print(f"Time taken A*: {end_time - start_time:.6f} seconds")
print(f"Memory usage A*: {memory_usage / (1024 * 1024):.2f} MB")
print()

"""# samples"""

samples = [[1,2,3,0,7,6,5,4,8],
[0,4,1,2,5,3,7,8,6],
[4,1,3,0,2,6,7,5,8],
[1,2,3,0,4,8,7,6,5],
[1,2,0,4,8,3,7,6,5],
[1,0,2,4,6,3,7,5,8],
[0,1,2,4,5,3,7,8,6],
[1,2,3,0,4,5,7,8,6],
[1,2,3,4,0,5,7,8,6],
[1,2,3,4,5,0,7,8,6],
[0,1,3,4,2,5,7,8,6],
[2,3,5,1,0,4,7,8,6],
[1,6,2,5,3,0,4,7,8],
[1,8,2,0,4,3,7,6,5],
[2,5,3,4,1,6,0,7,8],
[1,2,3,4,6,8,7,5,0],
[1,6,2,5,7,3,0,4,8],
[0,4,1,5,3,2,7,8,6],
[0,5,2,1,8,3,4,7,6],
[1,2,3,0,4,6,7,5,8],
[1,3,5,7,2,6,8,0,4],
[4,1,2,3,0,6,5,7,8],
[4,3,1,0,7,2,8,5,6],
[5,2,1,4,8,3,7,6,0],
[2,0,8,1,3,5,4,6,7],
[3,5,6,1,4,8,0,7,2],
[1,0,2,7,5,4,8,6,3],
[5,1,8,2,7,3,4,0,6],
[4,3,0,6,1,8,2,7,5],
[2,4,3,1,6,5,8,0,7],
[1,2,3,6,4,5,7,8,0],
[3,1,2,4,5,6,7,8,0],
[1,2,3,4,8,7,6,5,0],
[1,3,2,5,4,6,7,8,0],
[1,4,2,6,5,8,7,3,0],
[2,1,3,4,5,6,8,7,0],
[2,3,1,6,5,4,8,7,0],
[2,3,1,6,4,5,7,8,0],
[1,2,3,6,5,4,8,7,0],
[1,2,3,6,5,4,0,8,7],
[4,5,3,2,8,0,6,7,1],
[4,5,3,2,1,0,8,7,6],
[1,2,4,3,5,0,8,7,6],
[1,2,4,3,5,8,7,0,6],
[2,1,3,4,5,8,7,0,6],
[1,3,5,8,7,0,6,2,4],
[4,3,1,6,5,8,0,2,7],
[7,0,4,8,5,1,6,3,2],
[8,7,2,1,5,0,4,6,3],
[8,3,5,6,4,2,1,0,7],
[1,6,4,0,3,5,8,2,7],
[6,3,8,5,4,1,7,2,0],
[5,8,7,1,4,6,3,0,2],
[2,8,5,3,6,1,7,0,4],
[8,7,6,5,4,3,2,1,0]]

"""# Test BFS

"""

bfs_time = []
bfs_mem = []
bfs_results = []
for i in samples:
    start_time = time.perf_counter()
    tracemalloc.start()
    a, b = bfs_8puzzle(i, [1,2,3,4,5,6,7,8,0])
    end_time = time.perf_counter()
    _, memory_usage = tracemalloc.get_traced_memory()
    bfs_time.append(end_time - start_time)
    bfs_mem.append(memory_usage / (1024 * 1024))
    bfs_results.append(a)

"""# Test IDS

"""

ids_time = []
ids_mem = []
ids_results = []
for i in samples:
    start_time = time.perf_counter()
    tracemalloc.start()
    a, b = ids_8puzzle(i, [1,2,3,4,5,6,7,8,0])
    end_time = time.perf_counter()
    _, memory_usage = tracemalloc.get_traced_memory()
    ids_time.append(end_time - start_time)
    ids_mem.append(memory_usage / (1024 * 1024))
    ids_results.append(a)

"""# Test UCS"""

ucs_time = []
ucs_mem = []
ucs_results = []
for i in samples:
    start_time = time.perf_counter()
    tracemalloc.start()
    a, b = ucs_8puzzle(i, [1,2,3,4,5,6,7,8,0])
    end_time = time.perf_counter()
    _, memory_usage = tracemalloc.get_traced_memory()
    ucs_time.append(end_time - start_time)
    ucs_mem.append(memory_usage / (1024 * 1024))
    ucs_results.append(a)

"""# Test A*"""

astar_time = []
astar_mem = []
astar_results = []
for i in samples:
    start_time = time.perf_counter()
    tracemalloc.start()
    a, b = astar_8puzzle(i, [1,2,3,4,5,6,7,8,0])
    end_time = time.perf_counter()
    _, memory_usage = tracemalloc.get_traced_memory()
    astar_time.append(end_time - start_time)
    astar_mem.append(memory_usage / (1024 * 1024))
    astar_results.append(a)

"""# Test DFS

"""

dfs_time = []
dfs_mem = []
dfs_results = []
for i in samples:
    start_time = time.perf_counter()
    tracemalloc.start()
    a, b = dfs_8puzzle(i, [1,2,3,4,5,6,7,8,0])
    end_time = time.perf_counter()
    _, memory_usage = tracemalloc.get_traced_memory()
    dfs_time.append(end_time - start_time)
    dfs_mem.append(memory_usage / (1024 * 1024))
    dfs_results.append(a)