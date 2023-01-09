# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Shang-Tse Chen (stchen@csie.ntu.edu.tw) on 03/03/2022
from distutils.archive_util import make_zipfile
from maze import *
from heapq import *
"""
This is the main entry point for HW1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)

def manhattanDist(start, end):
    # manhattan distance
    return abs(start[0] - end[0]) + abs(start[1] - end[1])

class State():
    def __init__(self, pos, is_eaten, g, h, prev):
        self.pos = pos
        self.is_eaten = is_eaten
        self.g = g
        self.h = h
        self.prev = prev
        self.dots_left = sum(1-x for x in is_eaten if x == 0)

'''
    State: 
        (x, y), (dots_not_eaten, ...), g, h, dots_left, prev,
        heuristic: cost*len + sum(dist)
        g: $len cost per move

    State_Manager: 
        [dots], fringe(heap[f, stateid]), [all_states], states
        paths_dict {(start, end): [route]}
        pathsWithDots_dict {(start, end): [is_conjuct]}
        states_closed: {((pos), (is_eaten)), ...}
        discarded: states_dict ((x, y), (is_eaten)) -> nodeid
        
        is_goal(stateid)
            return dots_left == 0
        update_h(stateid)
        make_state((x,y), prev) -> (f, stateid) or None when already exists
        get_path(state_id)
        get_closestDot(start, (is_eaten))
        get_shortestPath(start, end) #using A* search
        
        
'''

class State_Manager():
    def __init__(self, maze: Maze):
        init_pos = maze.getStart()
        self.maze = maze
        self.dots = maze.getObjectives()
        self.n_dots = len(self.dots)
        self.states = [State(init_pos, (0,)*self.n_dots, 0, 0, -1)]
        self.states_Closed = set()
        self.paths_dict = {}
        self.pathsWithDots_dict = {}
        self.update_h(0)

    def is_goal(self, state_id):
        return self.states[state_id].dots_left == 0

    def is_Closed(self, state_id):
        # node: ((pos, (is_eaten))
        state = self.states[state_id]
        return (state.pos, state.is_eaten) in self.states_Closed

    def add_Closed(self, state_id):
        state = self.states[state_id]
        self.states_Closed.add((state.pos, state.is_eaten))
    
    def make_state(self, pos, prev):
        prev_state = self.states[prev]
        is_eaten = list(self.states[prev].is_eaten)
        for i in range(self.n_dots):
            if pos == self.dots[i]:
                is_eaten[i] = 1
                break
        state = State(pos, tuple(is_eaten), prev_state.g + self.n_dots,
            prev_state.h, prev)
        state_id = len(self.states)
        self.states.append(state)
        self.update_h(state_id)
        return (state.g + state.h, state_id)
        
    def update_h(self, state_id):
        # sum of manhattan distance to all the remaining dots
        pos = self.states[state_id].pos
        h = 0
        for i in range(self.n_dots):
            if self.states[state_id].is_eaten[i]:
                continue
            h += manhattanDist(pos, self.dots[i])
        self.states[state_id].h = h

    def get_path(self, state_id):
        ret = []
        while state_id != -1:
            ret.append(self.states[state_id].pos)
            state_id = self.states[state_id].prev
        ret.reverse()
        return ret
    
    def develop(self, state_id):
        # may contain duplicate nodes that is closed, 
        # need to check whether the child is closed or not
        row, col = self.states[state_id].pos
        children_pos = self.maze.getNeighbors(row, col)
        return [self.make_state(pos, state_id) for pos in children_pos]
    
    def get_shortestPath(self, start, end, ub_cost = -1):
        if start == end:
            return [start]
        if (start, end) in self.paths_dict:
            return self.paths_dict[(start, end)]
        if (end, start) in self.paths_dict:
            li = self.paths_dict[(end, start)].copy()
            li.reverse()
            self.paths_dict[(start, end)] = li
            return li
        states, hp = [(start, 0, -1)], [(manhattanDist(start, end), 0)]
        # states: [(pos, g, prev), ...], hp: [(f, id)], closed: {pos, ...}
        Closed = set()
        while len(hp):
            state_id = heappop(hp)[1]
            pos, cost, prev = states[state_id]
            if ub_cost > 0 and cost >= ub_cost:
                return None
            if pos in Closed:
                continue
            if pos == end:
                # finish
                ret = []
                while state_id != -1:
                    ret.append(states[state_id][0])
                    state_id = states[state_id][2]
                ret.reverse()
                self.paths_dict[(start, end)] = ret
                return ret
            Closed.add(pos)
            for next in self.maze.getNeighbors(pos[0], pos[1]):
                new_id, new_cost = len(states), cost+1 + manhattanDist(next, end)
                if ub_cost > 0 and new_cost >= ub_cost:
                    continue
                states.append((next, cost+1, state_id))
                heappush(hp, (cost+1 + manhattanDist(next, end), new_id))
        return None

    def get_shortestPathwithDots(self, start, end):
        ret = [0,] * self.n_dots
        if (start, end) in self.pathsWithDots_dict:
            ret = self.pathsWithDots_dict[(start, end)]
        path = self.get_shortestPath(start, end)
        for x in path:
            if x in self.dots:
                ret[self.dots.index(x)] = 1
        self.pathsWithDots_dict[(start, end)] = self.pathsWithDots_dict[(end, start)] = ret
        return ret

    def get_closestDot(self, pos, is_eaten):
        ret = (-1, -1); minval = 1e10; ub_cost = -1
        for i in range(len(is_eaten)):
            if is_eaten[i]:
                continue
            a = self.get_shortestPath(pos, self.dots[i], ub_cost)
            if a != None and len(a) < minval:
                ub_cost = minval = len(a)
                ret = self.dots[i]
        return ret
    

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)

def bfs(maze: Maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    states, parents, dot = [maze.getStart()], [-1], maze.getObjectives()[0]
    start_id = 0

    while start_id < len(states):
        state = states[start_id]
        if state == dot:
            state_id = start_id
            ret = []
            while state_id >= 0:
                ret.append(states[state_id])
                state_id = parents[state_id]
            ret.reverse()
            return ret
        next_states = maze.getNeighbors(state[0], state[1])
        for next_state in next_states:
            if next_state not in states:
                states.append(next_state)
                parents.append(start_id)
        start_id += 1

def astar(maze: Maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples conta(ining the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    sm = State_Manager(maze)
    return sm.get_shortestPath(maze.getStart(), maze.getObjectives()[0])

def astar_corner(maze: Maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    # TODO: Write your code here
    sm, hp = State_Manager(maze), []
    heappush(hp, (sm.states[0].g + sm.states[0].h, 0))

    while len(hp):
        state_id = heappop(hp)[1]
        if sm.is_goal(state_id):
            return sm.get_path(state_id)
        if not sm.is_Closed((state_id)):
            sm.add_Closed(state_id)
            children = sm.develop(state_id)
            for child in children:
                if sm.is_Closed(child[1]):
                    continue
                heappush(hp, child)
    return []


def astar_multi(maze: Maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    # applies DP with Priority queue
    init_pos, dots, is_eaten = maze.getStart(), maze.getObjectives(), [0,]*len(maze.getObjectives())
    sm, states, hp = State_Manager(maze), [(0, is_eaten.copy(), init_pos, -1)], []
    isClosed = set()

    # states [((0)total_cost, (1)is_eaten, (2)last_dot(pos), (3)last_state_id), ....]
    # hp: [(total_cost, state_id), ...]
    # isClosed {(is_eaten, last_dot)}

    isClosed.add((tuple(is_eaten), init_pos))

    for i in range(len(is_eaten)):
        path, state_id = sm.get_shortestPath(init_pos, dots[i]), len(states)
        state = (len(path), is_eaten.copy(), dots[i], 0)
        state[1][i] = 1
        states.append(state)
        heappush(hp, (state[0], state_id))

    while len(hp) > 0:
        state_id = heappop(hp)[1]
        state = states[state_id]
        if (tuple(state[1]), state[2]) in isClosed:
            continue
        if sum(state[1]) == len(dots):
            # find the goal
            ret = []
            while state_id > 0:
                prev_id = states[state_id][3]
                ret.extend(sm.get_shortestPath(states[state_id][2], states[prev_id][2])[:-1])
                state_id = prev_id
            ret.append(init_pos)
            ret.reverse()
            return ret

        isClosed.add((tuple(state[1]), state[2]))
        for i in range(len(dots)):
            if state[1][i] == 1:
                # has eaten
                continue
            
            new_path, new_state_id = sm.get_shortestPath(state[2], dots[i]), len(states)
            new_state = (len(new_path) + state[0], state[1].copy(), dots[i], state_id)
            new_state[1][i] = 1
            
            is_conj = sm.get_shortestPathwithDots(state[2], dots[i])         
            for i in range(len(dots)):
                if is_conj[i]:
                    new_state[1][i] = 1

            if (tuple(new_state[1]), dots[i]) in isClosed:
                continue
            states.append(new_state)
            heappush(hp, (new_state[0], new_state_id))
    return []



def fast(maze: Maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    pos, dots, is_eaten = maze.getStart(), maze.getObjectives(), [0,]*len(maze.getObjectives())
    sm = State_Manager(maze)
    ret = [pos]
    for i in range(len(is_eaten)):
        end = sm.get_closestDot(pos, is_eaten)
        ret = ret + sm.get_shortestPath(pos, end)[1:]
        is_eaten[dots.index(end)] = 1
        pos = end
    return ret
