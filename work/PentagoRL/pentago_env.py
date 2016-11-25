# -*- coding: utf-8 -*-
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

INFO_KEY = "reason"

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

        self.episode = 0
               
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
        self.episode += 1

        if self.episode % 1000 == 0:
            print(self.board)

        self.player = 2 if self.opponent_starts else 1
        self.opponent = 2 if self.player == 1 else 1 

        self.board = np.zeros((self.size, self.size), dtype=np.uint8)
        
        if self.opponent_starts:
            _ = self.make_opponent_move()

        return self.board


    def _step(self, action):
        # Take player action
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        legal_move = self.play_action(action, self.player)
        if not legal_move:
            return self.board, self.illegal_move_reward, True, {INFO_KEY: "Player {}: illegal move".format(self.player)}

        player_won = self.has_player_won(self.player)
        if player_won:
            return self.board, self.win_reward, True, {INFO_KEY: "Player {}: won".format(self.player)}
        opponent_won = self.has_player_won(self.opponent) 
        if opponent_won:
            return self.board, self.loose_reward, True, {INFO_KEY: "Player {}: lost".format(self.player)}
        if player_won and opponent_won:
            return self.board, self.tie_reward, True, {INFO_KEY: "Player {}: tied".format(self.player)}

        # Take opponent action
        legal_move = self.make_opponent_move()
        if not legal_move: # Note: tie reward!!!
            return self.board, self.tie_move_reward, True, {INFO_KEY: "Player {}: illegal move".format(self.opponent)}

        player_won = self.has_player_won(self.player)
        if player_won:
            return self.board, self.win_reward, True, {INFO_KEY: "Player {}: lost".format(self.opponent)}
        opponent_won = self.has_player_won(self.opponent) 
        if opponent_won:
            return self.board, self.loose_reward, True, {INFO_KEY: "Player {}: won".format(self.opponent)}
        if player_won and opponent_won:
            return self.board, self.tie_reward, True, {INFO_KEY: "Player {}: tied".format(self.opponent)}

        # Return non-terminal state 
        return self.board, self.legal_move_reward, False, {INFO_KEY: "Player {}: legal move".format(self.player)}

        
    def _render(self, mode='human', close=False):
        #print(self.board)
        return


    def make_opponent_move(self):
        if self.opponent_policy == None:
            empty_squares = np.argwhere(self.board == 0)
            nb_legal_moves = len(empty_squares)
            assert nb_legal_moves > 0, "No possible legal moves for opponent"
            square_idx = self.np_random.randint(nb_legal_moves)
            x = empty_squares[square_idx][0]
            y = empty_squares[square_idx][1]
            quadrant = self.np_random.randint(2)
            clockwise = self.np_random.randint(2)
            return self.play_move(x, y, quadrant, clockwise, self.opponent)
        else:
            action = opponent_policy.get_action()
            return self.play_action(action, self.opponent)      
            
        
    def play_action(self, action, player):
        # Lookup move details
        move = self.action_map[action]
        ((x,y), quadrant, clockwise) = move
        return self.play_move(x, y, quadrant, clockwise, player)


    def play_move(self, x, y, quadrant, clockwise, player):
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