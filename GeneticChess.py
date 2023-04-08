import argparse
import io
import math
import numpy as np
import random
from stockfish import Stockfish, StockfishException


PIECES = ["", "P", "N", "B", "R", "Q", "p", "n", "b", "r", "q", "K", "k"]
PROBS = [32/62, 8/62, 2/62, 2/62, 2/62, 1/62, 8/62, 2/62, 2/62, 2/62, 1/62]
assert math.isclose(sum(PROBS), 1)
args = None
engine = None
# Composed from ASCII art archive. Credit to Brent James Benton for the knight.
dna = r"""
                                 |\_
O       o O       o O       o   /  .\_
| O   o | | O   o | | O   o |  |   ___)
| | O | | | | O | | | | O | |  |    \
| o   O | | o   O | | o   O |  |  =  |
o       O o       O o       O  /_____\
                              [_______]
"""


def board_to_fen(board):
    with io.StringIO() as s:
        for row in board:
            empty = 0
            for cell in row:
                if cell > 0:
                    if empty > 0:
                        s.write(str(empty))
                        empty = 0
                    s.write(PIECES[cell])
                else:
                    empty += 1
            if empty > 0:
                s.write(str(empty))
            s.write('/')
        s.seek(s.tell() - 1)
        s.write(' w - - 0 1')
        return s.getvalue()


def remove_backrank_pawns(board):
    for i in range(8):
        if PIECES[board[0, i]].lower() == "p":
            board[0, i] = 0
        if PIECES[board[7, i]].lower() == "p":
            board[7, i] = 0


def evaluate(fen):
    global engine
    if not engine.is_fen_valid(fen):
        return None
    try:
        engine.set_fen_position(fen)
        val = engine.get_evaluation()
    except StockfishException:
        engine = Stockfish(path=args.stockfish, depth=args.depth)
        return None
    if val["type"] != "cp":
        return None
    if val["value"] == 0:
        return None
    return val["value"] / 100


def randomize():
    val = None
    while val is None:
        board = np.random.choice(11, p=PROBS, size=(8, 8))
        kings = np.random.randint(8, size=(2, 2))
        board[kings[0, 0], kings[0, 1]] = 11
        board[kings[1, 0], kings[1, 1]] = 12
        remove_backrank_pawns(board)
        fen = board_to_fen(board)
        val = evaluate(fen)
    return board, kings, fen, val


def mutate(board, kings):
    val = None
    while val is None:
        mask = np.random.binomial(1, p=args.mutation_rate, size=(8, 8))
        delta = np.random.choice(11, p=PROBS, size=(8, 8)) * mask
        board_new = board * (1 - mask) + delta
        kings_new = kings.copy()
        if board_new[kings[0, 0], kings[0, 1]] != 11:
            kings_new[0] = np.random.randint(8, size=2)
            board_new[kings_new[0, 0], kings_new[0, 1]] = 11
        if board_new[kings[1, 0], kings[1, 1]] != 12:
            kings_new[1] = np.random.randint(8, size=2)
            board_new[kings_new[1, 0], kings_new[1, 1]] = 12
        remove_backrank_pawns(board_new)
        fen = board_to_fen(board_new)
        val = evaluate(fen)
    return board_new, kings_new, fen, val


def evolve():
    best = randomize()
    board, kings, fen, val = best
    print()
    print(f"[Generation 0] [Evaluation {val}] [FEN {fen}]")
    for gen in range(args.generations):
        pop = [best]
        for i in range(args.population_size - 1):
            fen_new = fen
            while fen_new == fen:
                board_new, kings_new, fen_new, val_new = mutate(board, kings)
            pop.append((board_new, kings_new, fen_new, val_new))
        random.shuffle(pop)
        best = list(sorted(enumerate(pop), key=lambda x: (abs(x[1][3]), x[0])))[0][1]
        board, kings, fen, val = best
        print(f"[Generation {gen + 1}] [Evaluation {val}] [FEN {fen}]")
    engine.set_depth(args.final_depth)
    engine.set_fen_position(fen)
    val = engine.get_evaluation()["value"] / 100
    return fen, val


def results(fen, val):
    print()
    print("RESULTS")
    print()
    engine.set_fen_position(fen)
    vis = engine.get_board_visual()
    print(vis)
    print("Evaluation:", val)
    print("FEN:", fen)
    print()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stockfish", required=True, help="path to stockfish binary", type=str)
    parser.add_argument("--depth", type=int, default=15)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--generations", type=int, default=15)
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--mutation-rate", type=float, default=2/64)
    parser.add_argument("--final-depth", type=int, default=25)
    args = parser.parse_args()
    return args


def header():
    print(dna)
    print("Genetic Chess")
    print("By Santiago Benoit")
    print()


def main():
    global args
    global engine
    header()
    args = get_args()
    print("Arguments:", args)
    engine = Stockfish(path=args.stockfish, depth=args.depth)
    seed = args.seed
    if seed is None:
        seed = np.random.randint(2**32)
    print("Random seed:", seed)
    random.seed(seed)
    np.random.seed(seed)
    fen, val = evolve()
    results(fen, val)


if __name__ == "__main__":
    main()
