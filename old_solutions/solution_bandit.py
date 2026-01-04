import numpy as np
import enum
import heapq

class Cell(enum.IntEnum):
    EMPTY = 0
    WALL = 1

class Action(enum.IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    GATHER = 4

class BanditWallace():
    def __init__(self):
        self.OEC=10
        self.start_pos=(3,3)
    
    def a_star_search(self, goal_heuristic):
        if goal_heuristic=='focused':
            goal_center=self.focused_search_origin
        elif goal_heuristic=='broad':
            if not self.gold_locations:
                goal_center=self.focused_search_origin
            else:
                y_center=np.mean([p[0] for p in self.gold_locations]); x_center=np.mean([p[1] for p in self.gold_locations])
                goal_center=(y_center, x_center)
        def heuristic(pos):
            return abs(goal_center[0]-pos[0])+abs(goal_center[1]-pos[1])
        #pq= [heursitic, cost, path, pos]
        pq=[(heuristic(self.current_pos)), 0, [], self.current_pos]
        visited_costs={self.current_pos:0}
        while pq:
            _,cost,path,pos=heapq.heappop(pq)
            if cost>visited_costs[pos]:
                continue 
            unvisited_maze_cell= not self.maze_map.get(self.current_pos, {}).get('visited', False)
            if goal_heuristic=="focused":
                unvisited_maze_cell=not self.maze_map.get(self.current_pos, {}).get("visited_by_focused_search", False)
            if unvisited_maze_cell:
                if goal_heuristic=focused 


