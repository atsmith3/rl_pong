import sys
import sdl2
import sdl2.ext
import random
from game import GameState

WHITE = sdl2.ext.Color(255, 255, 255)
RED = sdl2.ext.Color(255, 0, 0)

def run():
    while running:
        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                running = False
                break
        sdl2.SDL_Delay(10)
        world.process()

if __name__ == "__main__":
    sys.exit(run())
