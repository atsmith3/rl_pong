import sys
import sdl2
import sdl2.ext
import random
from game import GameState

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
    def __init__(self, world, game):
        factory = sdl2.ext.SpriteFactory(sdl2.ext.SOFTWARE)
        sp_paddle = factory.from_color(WHITE, size=(1, int(game.paddle_h*200)))
        self.paddle = Player(world, sp_paddle, 199, int(game.paddle_y*200))
        sp_ball = factory.from_color(RED, size=(2,2))
        self.ball = Ball(world, sp_ball, int(game.ball_x*200), int(game.ball_y*200))

    def update(self, game):
        self.paddle.sprite.position = (199, int(game.paddle_y*200))
        self.ball.sprite.position = (int(game.ball_x*200), int(game.ball_y*200))

def run():
    running = True
    sdl2.ext.init()
    window = sdl2.ext.Window("Pong", size=(200, 200))
    window.show()
    world = sdl2.ext.World()
    spriterenderer = SoftwareRenderer(window)
    world.add_system(spriterenderer)

    game = GameState()
    game.paddle_h = 1.0
    game_gui = GameGUI(world,game)

    while running:
        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                running = False
                break
        game.update(1) # Do Nothing
        game_gui.update(game)
        if game.over:
            pass
            #running = False
            #break
        sdl2.SDL_Delay(10)
        world.process()

if __name__ == "__main__":
    sys.exit(run())
