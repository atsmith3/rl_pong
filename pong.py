import sys
import sdl2
import sdl2.ext
import random
import time
from game import GameState
from pong_agent import SLPongAgent

WHITE = sdl2.ext.Color(255, 255, 255)
RED = sdl2.ext.Color(255, 0, 0)


class SoftwareRenderer(sdl2.ext.SoftwareSpriteRenderSystem):

    def __init__(self, window):
        super(SoftwareRenderer, self).__init__(window)

    def render(self, components):
        sdl2.ext.fill(self.surface, sdl2.ext.Color(0, 0, 0))
        super(SoftwareRenderer, self).render(components)


class Player(sdl2.ext.Entity):

    def __init__(self, world, sprite, posx=0, posy=0):
        self.sprite = sprite
        self.sprite.position = posx, posy


class Ball(sdl2.ext.Entity):

    def __init__(self, world, sprite, posx=0, posy=0):
        self.sprite = sprite
        self.sprite.position = posx, posy


class GameGUI:

    def __init__(self, game):
        sdl2.ext.init()
        self.window = sdl2.ext.Window("Pong", size=(200, 200))
        self.window.show()
        self.world = sdl2.ext.World()
        self.spriterenderer = SoftwareRenderer(self.window)
        self.world.add_system(self.spriterenderer)
        self.factory = sdl2.ext.SpriteFactory(sdl2.ext.SOFTWARE)
        self.sp_paddle = self.factory.from_color(
            WHITE, size=(1, int(game.paddle_h * 200)))
        self.paddle = Player(self.world, self.sp_paddle, 199,
                             int(game.paddle_y * 200))
        self.sp_ball = self.factory.from_color(RED, size=(2, 2))
        self.ball = Ball(self.world, self.sp_ball, int(game.ball_x * 200),
                         int(game.ball_y * 200))

    def update(self, game):
        ret = True
        self.paddle.sprite.position = (199, int(game.paddle_y * 200))
        self.ball.sprite.position = (int(game.ball_x * 200),
                                     int(game.ball_y * 200))
        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                ret = False
        sdl2.SDL_Delay(1)
        self.world.process()
        return ret


def run(gui: bool = False):
    running = True

    game = GameState()
    if gui:
        game_gui = GameGUI(game)

    agent = SLPongAgent("pong_agent.pth")
    total_inference: int = 0
    total_time_ns: int = 0

    while running:
        start_time = time.time_ns()
        action = agent.eval(game.get_context())
        end_time = time.time_ns()
        total_time_ns += end_time - start_time
        total_inference += 1
        #print(f"SLAgentAction: {action}")
        game.update(action)
        if gui:
            if not game_gui.update(game):
                running = False
        if game.over:
            print(f"Score = {game.score}")
            print(
                f"Total Inferences: {total_inference}; {float(total_time_ns)/float(total_inference):.3f} ns/inf"
            )
            running = False
            break


if __name__ == "__main__":
    sys.exit(run(True))
