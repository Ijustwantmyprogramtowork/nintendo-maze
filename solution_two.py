from maze import direction_to_dxy

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

    def act(self):
        self._update_map(obs)


    def _update_map(self,obs):
        """_summary_

        Parameters
        ----------
        obs : dict of actions with position (y,x), what there is around it ( wall or empty ), if the position has gold
            _description_
        """
        y,x,top,bottom,left,right,has_gold=obs
        self.current_pos=(y,x)
        if self.current_pos not in self.maze_map: self.maze_map[self.current_pos]={}
        self.maze_map[self.current_pos].update({'type':'empty', 'visited':True, 'has_gold':has_gold})
        if has_gold and self.current_pos not in self.gold_locations:
            path=_find_path(self.current_pos, self.start_pos)



    def _find_path(self,current_pos, start_pos):
        """_summary_

        Parameters
        ----------
        current_pos : tuple, current position where there is gold
        start_pos : tuple, start position of wallace
            _description_
        """

        