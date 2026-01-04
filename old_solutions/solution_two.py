from maze import direction_to_dxy
import collections
import enum
import numpy as np
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

class Wallace():
    def __init__(self):
        self.action_plan= []
        self.current_pos=(3,3)
        self.start_pos=(3,3)

        self.search_mode='BROAD'
        self.focused_search_origin=None
        self.FOCUSED_SEARCH_RADIUS= 4 # A définir en fonction du labyrinthe

        self.maze_map={}
        self.last_target_pos=None
        self.gold_locations={}
        self.total_gather=0

        self.UCB_C=2.0
        self.OEV=10.0
        self.action_to_dxy = {Action.UP: (-1, 0), Action.DOWN: (1, 0), Action.LEFT: (0, -1), Action.RIGHT: (0, 1)}
        self.old_pos=None
        self.is_stuck=False


    def act(self, obs, gold_received, done):
        """
        This function allows to predict the next action step for wallace in the labyrinth
        Parameters
        ----------
        obs : dict of actions with position (y,x), what there is around it ( wall or empty ), if the position has gold
        gold_received: float, value determined by the euclidian distance between the gold and the center of the gold cluster calculated in maze.py
        done: bool, if the episode is finished or not

        returns Action
        """
        self._update_map(obs)
        if self.old_pos==self.current_pos:
            self.is_stuck=True
            # print("Is stuck here",self.current_pos)
        else:
            self.is_stuck=False
        if self.maze_map.get(self.current_pos, {}).get('has_gold', False):
            if self.current_pos in self.gold_locations and self.gold_locations[self.current_pos]['visits']==0:
                self.initiate_focus_search()
                # if not done:
                #     return Action.GATHER # Au final on a un pb avec le plan non pas bon plan et on perd du temps
        if done:
            print("gold received:", gold_received)
            self._process_reward(gold_received)
            self.action_plan=[]
            self.search_mode="BROAD"
            return None
        if not self.action_plan:
            self._make_new_plan()
        if self.action_plan: 
            action=self.action_plan.pop(0)
            # print(action)
            if action==Action.GATHER:
                self.last_target_pos=self.current_pos
            if self.is_stuck==True and action != Action.GATHER:
                self._make_new_plan() # A voir si ca marche ça 
            
            self.old_pos=self.current_pos
            # if self.is_stuck:
                # print(f"""
                #       Action is {action} and is leading to {self.current_pos + self.action_to_dxy[action]} where there is {obs}
                #       """)
            return action
        else:
            return np.random.choice(list(self.action_to_dxy.keys()))

    def initiate_focus_search(self):
        """Initiates focus search
        """
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
            self.total_gather+=1
        self.last_target_pos=None

    def _update_map(self,obs):
        """_summary_

        Parameters
        ----------
        obs : dict of actions with position (y,x), what there is around it ( wall or empty ), if the position has gold
            _description_
        """
        y,x,top,left,right,bottom,has_gold=obs
        self.current_pos=(y,x)
        if self.current_pos not in self.maze_map: 
            self.maze_map[self.current_pos]={}
        self.maze_map[self.current_pos].update({'type':'empty', 'visited':True, 'has_gold':has_gold})
        if has_gold and self.current_pos not in self.gold_locations:
            path=self._find_path(self.current_pos, self.start_pos)
            if path is not None:
                self.gold_locations[self.current_pos]={'visits': 0, 'value':0,'path_cost':len(path), 'path': path }
        neighbours={(y-1,x):top, (y+1,x):bottom, (y,x-1):left, (y,x+1):right}
        for pos, cell_type in neighbours.items():
            if pos not in self.maze_map:
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
        visited={start_pos}
        q=collections.deque([(start_pos, [])])
        path=[]
        while q:
            (y,x), path=q.popleft()
            if (y,x)==end_pos:
                return path
            
            for action, (dy,dx) in self.action_to_dxy.items():
                neighbour_pos=(y+dy, x+dx)
                if neighbour_pos not in visited and self.maze_map.get(neighbour_pos, {}).get("type", None)!="wall":
                    visited.add(neighbour_pos)
                    q.append((neighbour_pos, path+[action]))
        return None
    


    def _make_new_plan(self):
        """assigne actions based on exploitation or exploration

        Returns
        -------
        returns None 
        """
        if self.search_mode=='FOCUSED':
            _,plan,_= self._run_astar_search(goal_heuristic='focused')
            if plan and len(plan)<self.FOCUSED_SEARCH_RADIUS:
                self.action_plan=plan
                return None
            else:
                self.search_mode= 'BROAD'
        exploit_target, exploit_score=self._evaluate_exploitation()
        explore_target, explore_path, explore_score=self._run_astar_search(goal_heuristic='broad')
        if exploit_score>=explore_score and exploit_score is not None:
            self.action_plan=list(self.gold_locations[exploit_target]['path'])+[Action.GATHER]
            print(self.action_plan)
        elif explore_path:
            self.action_plan=explore_path
        elif exploit_target is not None:
            self.action_plan=list(self.gold_locations[exploit_target]['path'])+ [Action.GATHER]
        else:
            self.action_plan=[np.random.choice(list(self.action_to_dxy.keys()))]


    def _run_astar_search(self, goal_heuristic):
        if goal_heuristic=='focused':
            goal_center=self.focused_search_origin
        elif goal_heuristic=='broad':
            if not self.gold_locations:
                goal_center=self.current_pos #Default
            else:
                y_centers=[p[0] for p in self.gold_locations.keys()]; x_centers=[p[1] for p in self.gold_locations.keys()]
                goal_center=(np.mean(y_centers), np.mean(x_centers))
        def heuristic(pos):
            return abs(pos[0]-goal_center[0])+abs(pos[1]-goal_center[1])
        pq= [(heuristic(self.current_pos), 0, [], self.current_pos)]
        visited_costs={self.current_pos: 0}
        while pq:
            _, cost, path, pos=heapq.heappop(pq)
            if cost>visited_costs[pos]:
                continue
            is_unvisited_maze_cell=not self.maze_map.get(pos,{}).get('visited', False)
            if goal_heuristic== 'focused':
                is_unvisited_maze_cell= not self.maze_map.get(pos,{}).get('visited_by_focused_search', False)
            if is_unvisited_maze_cell:
                if goal_heuristic=='focused':
                    if pos not in self.maze_map:
                        self.maze_map[pos]={}
                    self.maze_map[pos]['visited_by_focused_search']=True
                score=self.OEV/(len(path) + 1) if path else float('inf')
                return pos, path, score
            for action, (dy,dx) in self.action_to_dxy.items():
                neighbour=(pos[0]+dy, pos[1]+dx)
                if self.maze_map.get(neighbour, {}).get('type')!= "wall":
                    new_cost=cost+1
                    if neighbour not in visited_costs or new_cost<visited_costs[neighbour]:
                        visited_costs[neighbour]=new_cost
                        priority=new_cost+heuristic(neighbour)
                        # print(f"From {pos} → {neighbour} via {action.name}")
                        # print(f"square {neighbour} type being {self.maze_map.get(neighbour,{}).get('type')}")
                        heapq.heappush(pq, (priority, new_cost, path+[action], neighbour))
        return None, None, -float('inf')
    


    def _evaluate_exploitation(self):
        if not self.gold_locations:
            return None, -float('inf')
        best_score, best_target= -float('inf'), None
        for pos, data in self.gold_locations.items():
            score= float('inf') if data['visits']==0 else (data['value']+self.UCB_C*np.sqrt(np.log(self.total_gather+1)/data['visits']))/( data['path_cost']+1)
            if score> best_score:
                best_score, best_target=score, pos
        if best_score== float('inf'):
            unvisited={p:d for p,d in self.gold_locations.items() if d['visits']==0}
            best_target=min(unvisited, key=lambda p : unvisited[p]['path_cost'])
        return best_target, best_score
        
    def get_custom_render_infos(self):
        render_list = []; [render_list.append((pos, (255, 0, 0))) for pos in self.gold_locations]
        if self.action_plan:
            path_pos = self.current_pos
            for action in self.action_plan:
                if action != Action.GATHER: dy, dx = self.action_to_dxy[action]; path_pos = (path_pos[0] + dy, path_pos[1] + dx); render_list.append((path_pos, (0, 255, 255)))
        return render_list


