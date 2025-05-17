import random
import numpy as np
import numpy.typing as npt
from typing import Dict, List, Any


class GameState:

    def __init__(self):
        self.ball_x = random.uniform(0.2, 0.8)
        self.ball_y = random.uniform(0.2, 0.8)
        self.ball_vx = random.uniform(-0.03, 0.03)
        self.ball_vy = random.uniform(-0.015, 0.015)
        self.paddle_y = 0.0
        self.paddle_h = 0.2
        self.paddle_vy = 0.04
        self.score = 0
        self.over = False

    def update(self, direction: int = 0):
        # Action Still = 1
        # Action Down = 2
        # Action Up = 0

        # Update Padde Position:
        if direction == 0:
            self.paddle_y = max(self.paddle_y - self.paddle_vy, 0.0)
        elif direction == 2:
            if self.paddle_y + self.paddle_h > 1.0:
                self.paddle_y = 1.0 - self.paddle_h
            else:
                self.paddle_y += self.paddle_vy

        # Update Ball Position and Velocity
        ball_x_n = self.ball_x + self.ball_vx
        ball_y_n = self.ball_y + self.ball_vy

        # Check for colisions with wall
        if ball_x_n < 0.0:
            ball_x_n = 0.0
            self.ball_vx = -self.ball_vx
        if ball_x_n > 1.0:
            if ball_y_n <= self.paddle_y and ball_y_n >= self.paddle_y + self.paddle_h:
                ball_x_n = 1.0
                self.ball_vx = random.uniform(-0.03, 0.0)
                self.ball_vy = random.uniform(-0.015, 0.015)
                self.score += 1
            else:
                self.over = True
                ball_x_n = 1.0
                self.ball_vx = -self.ball_vx
        if ball_y_n < 0.0:
            ball_y_n = 0.0
            self.ball_vy = -self.ball_vy
        if ball_y_n > 1.0:
            ball_y_n = 1.0
            self.ball_vy = -self.ball_vy

        self.ball_x = ball_x_n
        self.ball_y = ball_y_n

    def get_context(self) -> npt.NDArray[np.float32]:
        return np.array([
            self.ball_x, self.ball_y, self.ball_vx, self.ball_vy, self.paddle_y
        ])
