import sys
import sdl2
import sdl2.ext
import random
import time
import argparse
import statistics
from typing import List
from game import GameState
from agents.sl_agent import SLPongAgent
from agents.dqn_agent import DQNPongAgent

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


def run(args):
    label = f"{args.batch_size}_{args.epochs}_{args.learning_rate}_{args.layers}_{args.dimension}"
    print(f"Beginning Training for {label}")

    #agent = SLPongAgent(args.layers, args.dimension)
    #agent.train(args.batch_size, args.epochs, args.learning_rate)
    agent = DQNPongAgent(args.layers, args.dimension)
    agent.train()

    running = True
    game = GameState()
    if args.gui:
        game_gui = GameGUI(game)

    total_inference: int = 0
    total_time_us: float = 0.0

    scores: List[int] = []

    for match in range(args.matches):
        running = True
        game.reset()
        while running:
            start_time = time.time_ns()
            action = agent.eval(game.get_context())
            end_time = time.time_ns()
            total_time_us += (end_time - start_time) / 1000.0
            total_inference += 1
            game.update(action)
            if args.gui:
                if not game_gui.update(game):
                    running = False
            if game.over:
                scores.append(game.score)
                print(f"[{match:>4}]Score = {game.score}")
                running = False

    print(f"Average Score: {statistics.mean(scores)}")
    print(
        f"Total Inferences: {total_inference}; {float(total_time_us)/float(total_inference):.3f} us/inf"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Supervised Learning Pong Agent")
    parser.add_argument("-g",
                        "--gui",
                        action="store_true",
                        default=False,
                        help="Enable GUI Mode (Runs slower), skips training")
    parser.add_argument("-t",
                        "--train",
                        type=str,
                        default="./expert_policy.txt",
                        help="Enable training from expert policy dataset")
    parser.add_argument("-b",
                        "--batch_size",
                        type=int,
                        default=100,
                        help="Batch Size if training enabled")
    parser.add_argument("-e",
                        "--epochs",
                        type=int,
                        default=500,
                        help="Training Epochs if training enabled")
    parser.add_argument("-lr",
                        "--learning_rate",
                        type=float,
                        default=1e-1,
                        help="Learning Rate if training enabled")
    parser.add_argument("-l",
                        "--layers",
                        type=int,
                        default=1,
                        help="Hidden layers")
    parser.add_argument("-d",
                        "--dimension",
                        type=int,
                        default=512,
                        help="Hidden layer dimension")
    parser.add_argument("-m",
                        "--matches",
                        type=int,
                        default=100,
                        help="Number of random games for benchmarking")
    args = parser.parse_args()

    sys.exit(run(args))
