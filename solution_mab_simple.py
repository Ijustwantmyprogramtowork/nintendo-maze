import numpy as np
import enum
import heapq
import collections

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
        # --- Core State & Planning ---
        self.action_plan = []
        self.current_pos = (3, 3)
        self.start_pos = (3, 3)

        # --- Search Mode & Focused Search State ---
        self.search_mode = 'BROAD'
        self.focused_search_origin = None
        self.FOCUSED_SEARCH_RADIUS = 4

        # --- Knowledge & Exploitation ---
        self.maze_map = {}
        self.last_target_pos = None
        self.gold_locations = {}
        self.total_gathers = 0

        # --- Hyperparameters & Helpers ---
        self.UCB_C = 2.0
        self.OPTIMISTIC_EXPLORATION_VALUE = 10.0
        self.action_to_dxy = {Action.UP: (-1, 0), Action.DOWN: (1, 0), Action.LEFT: (0, -1), Action.RIGHT: (0, 1)}
        self.phase="EXPLORE" # Or EXPLOIT
        self.MIN_GOLDS=4
        self.current_target=None

    def act(self, obs, gold_received, done):
        """The main function called at each step."""
        self._update_map(obs)
        if done:
            self._process_reward(gold_received)
            self.action_plan = []; self.search_mode = 'BROAD'
            return None
        if len(self.gold_locations.keys())>=self.MIN_GOLDS:
            self.phase="EXPLOIT"
            #Forcing Gathering to start mab
            action=Action.GATHER
            return action

        if self.phase=="EXPLOIT" and self.current_target is not None:
            #Fllow commited plan and not redo it if problem
            pass
        elif not self.action_plan: 
            self._make_new_plan()
        

        if self.action_plan:
            action = self.action_plan.pop(0)
            if action == Action.GATHER and self.phase=="EXPLOIT": 
                self.last_target_pos = self.current_pos
            elif action==Action.GATHER and self.phase=="EXPLORE":
                return np.random.choice(list(self.action_to_dxy.keys()))
            return action
        else:
            return np.random.choice(list(self.action_to_dxy.keys()))
    
    def _run_astar_search(self, goal_heuristic):
        """
        Performs an A* search to find the best reachable, unvisited cell.
        The heuristic guides the search towards the desired area.
        """
        if goal_heuristic == 'broad':
            if not self.gold_locations: goal_center = self.current_pos # Default search goal
            else:
                y_coords = [p[0] for p in self.gold_locations.keys()]; x_coords = [p[1] for p in self.gold_locations.keys()]
                goal_center = (np.mean(y_coords), np.mean(x_coords))
        elif goal_heuristic == 'focused':
            goal_center = self.focused_search_origin
        
        def heuristic(pos): return abs(pos[0] - goal_center[0]) + abs(pos[1] - goal_center[1])

        pq = [(heuristic(self.current_pos), 0, [], self.current_pos)] # (priority, cost, path, pos)
        visited_costs = {self.current_pos: 0}

        while pq:
            _, cost, path, pos = heapq.heappop(pq)
            if cost > visited_costs[pos]: continue

            is_unvisited_maze_cell = not self.maze_map.get(pos, {}).get('visited', False)
            if goal_heuristic == 'focused':
                is_unvisited_maze_cell = not self.maze_map.get(pos, {}).get('visited_by_focused_search', False)
            
            if is_unvisited_maze_cell:
                if goal_heuristic == 'focused':
                    if pos not in self.maze_map: self.maze_map[pos] = {}
                    self.maze_map[pos]['visited_by_focused_search'] = True
                score = self.OPTIMISTIC_EXPLORATION_VALUE / (len(path) + 1) if path else float('inf')
                return pos, path, score

            for action, (dy, dx) in self.action_to_dxy.items():
                neighbor = (pos[0] + dy, pos[1] + dx)
                if self.maze_map.get(neighbor, {}).get('type') != 'wall':
                    new_cost = cost + 1
                    if neighbor not in visited_costs or new_cost < visited_costs[neighbor]:
                        visited_costs[neighbor] = new_cost
                        priority = new_cost + heuristic(neighbor)
                        heapq.heappush(pq, (priority, new_cost, path + [action], neighbor))
        return None, None, -float('inf')
    
    def _make_new_plan(self):
        exploit_target, exploit_score = self._evaluate_exploitation()
        explore_target, explore_plan, explore_score = self._run_astar_search(goal_heuristic='broad')
        if explore_plan and self.phase=="EXPLORE":
            self.action_plan=explore_plan
        elif exploit_target is not None and self.phase=="EXPLOIT":
            self.current_target=exploit_target
            self.action_plan = list(self.gold_locations[exploit_target]['path']) + [Action.GATHER]
    
    def _initiate_focused_search(self):
        """Triggers the 'circling' behavior if not already focused on this spot."""
        if self.focused_search_origin != self.current_pos:
            self.search_mode = 'FOCUSED'
            self.action_plan = []
            self.focused_search_origin = self.current_pos
            # Since we just arrived, we mark the origin as "visited" for the purpose of the focused search
            if self.focused_search_origin not in self.maze_map: self.maze_map[self.focused_search_origin] = {}
            self.maze_map[self.focused_search_origin]['visited_by_focused_search'] = True
    def _update_map(self, obs):
        y, x, top, left, right, bottom, has_gold = obs
        #to remove
        self.current_pos = (y, x)
        if self.current_pos not in self.maze_map: 
            self.maze_map[self.current_pos] = {}
        self.maze_map[self.current_pos].update({'type': 'empty', 'visited': True, 'has_gold': has_gold})
        if has_gold and self.current_pos not in self.gold_locations:
            path_to_gold = self._find_path(self.start_pos, self.current_pos)  
            if path_to_gold is not None: 
                self.gold_locations[self.current_pos] = {'visits': 0, 'value': 0, 'path_cost': len(path_to_gold), 'path': path_to_gold}
        neighbors = {(y - 1, x): top, (y + 1, x): bottom, (y, x - 1): left, (y, x + 1): right}
        for pos, cell_type in neighbors.items():
            if pos not in self.maze_map:
                self.maze_map[pos] = {'type': 'wall' if cell_type == Cell.WALL else 'empty', 'visited': False}
    def _process_reward(self, gold_received):
        if self.last_target_pos and self.last_target_pos in self.gold_locations:
            loc_data = self.gold_locations[self.last_target_pos]
            loc_data['value'] = (loc_data['value'] * loc_data['visits'] + gold_received) / (loc_data['visits'] + 1)
            loc_data['visits'] += 1; self.total_gathers += 1
        self.last_target_pos = None
        self.current_target = None
    def _find_path(self, start_pos, end_pos):
        q=collections.deque([(start_pos, [])])
        visited={start_pos}
        while q:
            (y,x), path=q.popleft()
            if (y,x)==end_pos:
                return path
            for action, (dy,dx) in self.action_to_dxy.items():
                neighbour=(y+dy, x+dx)
                if neighbour not in visited and self.maze_map.get(neighbour, {}).get('type')!="wall":
                    q.append((neighbour, path+[action]))
                    visited.add(neighbour)
            return None
        
    def _evaluate_exploitation(self):
        if not self.gold_locations:
            return None, -float('inf')
        best_score, best_target = -float('inf'), None
        for pos, data in self.gold_locations.items():
            score = float('inf') if data['visits'] == 0 else (data['value'] + self.UCB_C * np.sqrt(np.log(self.total_gathers + 1) / data['visits'])) / (data['path_cost'] + 1)
            if score > best_score:
                best_score, best_target = score, pos
        if best_score == float('inf'):
            unvisited = {p: d for p, d in self.gold_locations.items() if d['visits'] == 0}
            best_target = min(unvisited, key=lambda p: unvisited[p]['path_cost'])
        return best_target, best_score



    
    def get_custom_render_infos(self):
        render_list = []; [render_list.append((pos, (255, 0, 0))) for pos in self.gold_locations]
        if self.search_mode=="BROAD":
            pixel_color=(0,125,125)
        elif self.search_mode=="FOCUSED":
            pixel_color=(125,125,0)
        render_list.append((self.current_pos, pixel_color))
        if self.action_plan:
            path_pos = self.current_pos
            for action in self.action_plan:
                if action != Action.GATHER: dy, dx = self.action_to_dxy[action]; path_pos = (path_pos[0] + dy, path_pos[1] + dx); render_list.append((path_pos, (0, 255, 255)))
        return render_list


    

            


