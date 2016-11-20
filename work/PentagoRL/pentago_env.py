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
    
    
    def __init__(self):
        # Constants
        self.size = 4
        self.halfsize = self.size / 2        
        self.loose_reward = -10
        self.illegal_move_reward = -5
        self.legal_move_reward = -1
        self.win_reward = 10
        
        #self.viewer = None
        
        # Size of board * 4 quadrants * 2 clock rotate directions
        moves = [((x,y), q, c) for x in range(self.size) for y in range(self.size) for q in range(4) for c in range(2)]
        self.action_map = {action: move for (action, move) in enumerate(moves)}
        self.action_space = spaces.Discrete(len(moves))
        
        # 3 "colors" pr square
        self.observation_space = spaces.Box(low=0, high=2, shape=(self.size, self.size))

        self._seed()
        self._reset()

        
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def _step(self, action):
        self.switch_player()

        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        move = self.action_map[action]
        legal_move = self.play_move(move)
        
        if not legal_move:
            return self.board, self.illegal_move_reward, True, "Player {}: illegal move".format(self.player)
                     
        if self.has_player_won(self.player):
            return self.board, self.win_reward, True, "Player {}: won".format(self.player)
        if self.has_player_won(self.opponent):
            return self.board, self.loose_reward, True, "Player {}: lost".format(self.player)
        
        return self.board, self.legal_move_reward, False, "Player {}: legal move".format(self.player)           


    def _reset(self):
        self.player = 2
        self.opponent = 2 if self.player == 1 else 1 
        self.board = np.zeros((self.size, self.size), dtype=np.uint8)
        return self.board

        
    def _render(self, mode='human', close=False):
        #print(self.board)
        return
            
        
    def play_move(self, move):
        ((x,y), quadrant, clockwise) = move
        
        if self.board[x, y] > 0:
            return False
        self.board[x, y]= self.player

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
        
        
    def switch_player(self):
        self.player = 2 if self.player == 1 else 1
        self.opponent = 2 if self.player == 1 else 1 

    
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
        
        
    def is_winseq(self, l):
        return np.all(np.equal(l, self.winseq))
