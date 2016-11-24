# -*- coding: utf-8 -*-
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class PentagoEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }
    
    
    def __init__(self, size, opponent_starts = False, opponent_policy = None):
        # Constants
        self.size = size
        self.halfsize = self.size / 2
        self.opponent_starts = opponent_starts
        self.opponent_policy = opponent_policy        

        self.loose_reward = -1
        self.illegal_move_reward = -.5
        self.legal_move_reward = -.1
        self.tie_reward = 0
        self.win_reward = 1
               
        # Board: size * 4 quadrants * 2 clock rotate directions
        moves = [((x,y), q, c) for x in range(self.size) for y in range(self.size) for q in range(4) for c in range(2)]        
        self.action_map = {action: move for (action, move) in enumerate(moves)}
        #self.moves_map = {move: action for (action, move in enumerate(moves)}
        
        # Spaces
        self.observation_space = spaces.Box(low=0, high=2, shape=(self.size, self.size)) # 3 "colors" pr square
        self.action_space = spaces.Discrete(len(moves))

        self._seed()
        self._reset()

        
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def _reset(self):
        self.player = 2 if self.opponent_starts else 1
        self.opponent = 2 if self.player == 1 else 1 

        self.board = np.zeros((self.size, self.size), dtype=np.uint8)
        
        if self.opponent_starts:
            _ = self.take_opponent_action()

        return self.board


    def _step(self, action):
        # Take player action
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        legal_move = self.play_move(action, self.player)
        if not legal_move:
            return self.board, self.illegal_move_reward, True, "Player {}: illegal move".format(self.player)

        player_won = self.has_player_won(self.player)
        if player_won:
            return self.board, self.win_reward, True, "Player {}: won".format(self.player)
        opponent_won = self.has_player_won(self.opponent) 
        if opponent_won:
            return self.board, self.loose_reward, True, "Player {}: lost".format(self.player)
        if player_won and opponent_won:
            return self.board, self.tie_reward, True, "Player {}: tied".format(self.player)

        # Take opponent action
        legal_move = self.take_opponent_action()
        if not legal_move: # Note: tie reward!!!
            return self.board, self.tie_move_reward, True, "Player {}: illegal move".format(self.opponent)

        player_won = self.has_player_won(self.player)
        if player_won:
            return self.board, self.win_reward, True, "Player {}: lost".format(self.opponent)
        opponent_won = self.has_player_won(self.opponent) 
        if opponent_won:
            return self.board, self.loose_reward, True, "Player {}: won".format(self.opponent)
        if player_won and opponent_won:
            return self.board, self.tie_reward, True, "Player {}: tied".format(self.opponent)

        # Return non-terminal state 
        return self.board, self.legal_move_reward, False, "Player {}: legal move".format(self.player)

        
    def _render(self, mode='human', close=False):
        #print(self.board)
        return


    def take_opponent_action(self):
        if self.opponent_policy == None:
            action = self.np_random.randint(self.action_space.n)
        else:
            action = opponent_policy.get_action()
        return = self.play_move(action, self.opponent)      
            
        
    def play_move(self, action, player):
        # Lookup move details
        move = self.action_map[action]
        ((x,y), quadrant, clockwise) = move

        # Check legal move
        if self.board[x, y] > 0:
            return False

        # Make move
        self.board[x, y] = player
        x_, y_= 0, 0
        if quadrant == 1 or quadrant == 3:
            x_ += self.halfsize
        if quadrant == 2 or quadrant == 3:
            y_ += self.halfsize
        oldboard = np.copy(self.board)  
        for i in range(self.halfsize):
            for j in range(self.halfsize):
                self.board[x_ + i, y_ + j] = \
                    oldboard[x_ + self.halfsize - 1 - j, y_ + i] if clockwise else \
                    oldboard[x_ + j][y_ + self.halfsize - 1 - i]

        return True

    
    def has_player_won(self, player):
        self.winseq = np.ones(self.size - 1) * player

        #check lines & columns
        for i in range(self.size):
            if self.is_winseq(self.board[i,:-1]) or self.is_winseq(self.board[i,1:]):
                return True
            if self.is_winseq(self.board[:-1,i]) or self.is_winseq(self.board[1:,i]):
                return True
                
        #check 2 main diagonals
        diag = self.board.diagonal()
        if self.is_winseq(diag[:-1]) or self.is_winseq(diag[1:]):
            return True
        flipped = np.fliplr(self.board)
        diag = flipped.diagonal()
        if self.is_winseq(diag[:-1]) or self.is_winseq(diag[1:]):
            return True
                
        #check the 4 small diagonals
        if self.is_winseq(self.board.diagonal(offset=1)) or self.is_winseq(self.board.diagonal(offset=-1)):
            return True
        if self.is_winseq(flipped.diagonal(offset=1)) or self.is_winseq(flipped.diagonal(offset=-1)):
            return True
            
        return False
        
        
    def is_winseq(self, seq):
        return np.all(np.equal(seq, self.winseq))