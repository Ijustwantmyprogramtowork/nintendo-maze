from maze import direction_to_dxy
import collections
import enum
import numpy as np


class Cell(enum.IntEnum):
    EMPTY = 0
    WALL = 1

class Action(enum.IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    GATHER = 4

class Wallace_2():
    def __init__(self):
        self.action_plan= []
        self.current_pos=(3,3)
        self.start_pos=(3,3)

        self.search_mode='BROAD'
        self.focused_search_origin=None
        self.FOCUSED_SEARCH_RADIUS= # A définir en fonction du labyrinthe

        self.maze_map={}
        self.last_target_pos=None
        self.gold_locations={}
        self.total_gather=0

        self.UCB_C=2.0
        self.OEV=10.0
        self.action_to_dxy = {Action.UP: (-1, 0), Action.DOWN: (1, 0), Action.LEFT: (0, -1), Action.RIGHT: (0, 1)}


    def act(self, obs, done, gold_received):
        self._update_map(obs)
        if self.maze_map.get(self.current_pos, {}).get('has_gold', False):
            if self.gold_locations[self.current_pos]['visits']==0:
                self.initiate_focus_search()
        if done:
            self._process_reward(gold_received)
            self.action_plan=[]
            self.search_mode="BROAD"
            return None
        if not self.action_plan:
            self._make_new_plan()

    def initiate_focus_search(self):
        if self.focused_search_origin!= self.current_pos:
            self.search_mode = 'FOCUSED'
            self.action_plan=[]
            self.focused_search_origin=self.current_pos
            if self.focused_search_origin not in self.maze_map:
                self.maze_map[self.focused_search_origin]={}
            self.maze_map[self.focused_search_origin]['visited_by_focused_search'] = True


    def _process_reward(self, gold_received):
        if self.last_target_pos and self.last_target_pos in self.gold_locations:
            loc_data=self.gold_locations[self.last_target_pos]
            loc_data['value']= (loc_data['value']*loc_data['visits'] + gold_received)/(loc_data['visits']+1)
            loc_data['visits']+=1
            self.total_gathers+=1
        self.last_target_pos=None

    def _update_map(self,obs):
        """_summary_

        Parameters
        ----------
        obs : dict of actions with position (y,x), what there is around it ( wall or empty ), if the position has gold
            _description_
        """
        y,x,top,bottom,left,right,has_gold=obs
        self.current_pos=(y,x)
        if self.current_pos not in self.maze_map: 
            self.maze_map[self.current_pos]={}
        self.maze_map[self.current_pos].update({'type':'empty', 'visited':True, 'has_gold':has_gold})
        if has_gold and self.current_pos not in self.gold_locations:
            path=self._find_path(self.current_pos, self.start_pos)
            if path is not None:
                self.gold_locations[self.current_pos]={'visits': 0, 'value':0,'path_cost':len(path), 'path': path }
        neighbours={top:(y-1,x), bottom:(y+1,x), left:(y,x-1), right:(y,x+1)}
        for cell_type,pos in neighbours.items():
            if pos not in self.maze_map():
                self.maze_map[pos]={'type': "wall" if cell_type==Cell.WALL else "empty", 'visited':False}



    def _find_path(self,start_pos, end_pos):
        """_summary_

        Parameters
        ----------
        end_pos : tuple, current position where there is gold
        start_pos : tuple, start position of wallace
        
        returns
        list
        the shortest path to find gold
        """
        parents={}
        visited={}
        q=collections.deque(start_pos)
        path=[]
        while q:
            (y,x)=q.popleft()
            if (y,x)==end_pos:
                while (y,x)!= start_pos:
                    path.append([y,x])
                    (y,x)=parents[(y,x)]
                return path
            
            for action, (dy,dx) in self.action_to_dxy.items():
                neighbour_pos=(y+dy, x+dx)
                if neighbour_pos not in visited and self.maze_map.get(neighbour_pos, {})!="wall":
                    parents[neighbour_pos]=(y,x)
                    visited.add(neighbour_pos)
                    q.append(neighbour_pos)
        return
    


    def _make_new_plan(self):
        if self.search_mode=='focused':
            self._run_astar_search(self)

    def _run_astar_search(self, goal_heuristic):
        if goal_heuristic=='focused':
            goal_center=self.focused_search_origin
        elif goal_heuristic=='broad':
            if not self.gold_locations:
                goal_center=self.current_pos #Default
            y_centers=[p[0] for p in self.gold_locations.keys()]; x_centers=[p[1] for p in self.gold_locations.keys()]
            goal_center=(np.mean(y_centers), np.mean(x_centers))
        def heuristic(pos):
            return abs(pos[0]-goal_center[0])+abs(pos[1]-goal_center[1])
        pq= [(heuristic(self.current_pos), 0, [], self.current_pos)]
        visited_costs={self.current_pos: 0}
        while pq:
            _, cost, path, pos=heapq.heappop
        