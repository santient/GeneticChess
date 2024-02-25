# Funky Chess Variant AKA Funky Freestyle Chess
# By Santiago Benoit
# 
# Purpose:
# - Create games that are diverse, unique, and interesting
# - No theory (practically speaking), thinking starts at the beginning
# - Likely asymmetric
# - No unfair advantage, balanced for both players
# - Gives both players many options for how to play
# - Can be played with standard chess set
# 
# Rules:
# - Set up boards like in "transcendental chess":
# -- Shuffle both backranks, asymmetry allowed, bishops on opposite colors
# - No castling rights
# - To deal with imbalance:
# -- Engine makes the move with evaluation closest to equal (or custom odds)
# -- All legal moves considered, ties broken randomly
# -- Engine continues making moves until evaluation is within tolerance
# -- At least 1 move will be made
# -- Players start at resulting position (including turn, en passant, half and fullmove clocks)
# - Gameplay can possibly start with either white or black, pieces off the board, or player in check!
# - Aside from the starting position, same rules as regular chess


import argparse
import chess
import numpy as np
import secrets
from stockfish import Stockfish


# version = "1.0.0"


def balance(fen, args):
    engine = Stockfish(path=args.engine, depth=args.depth, parameters={"Threads": args.threads, "Hash": args.hash})
    engine.set_fen_position(fen)
    board = chess.Board(fen)
    error = float("inf")
    print(fen)
    while error > args.tolerance:
        legal = list(board.legal_moves)
        moves = engine.get_top_moves(len(legal))
        moves = [(move["Move"], move["Centipawn"] / 100) for move in moves if move["Centipawn"] is not None]
        np.random.shuffle(moves)
        move, val = min(moves, key=lambda x: abs(x[1] - args.odds))
        error = abs(val - args.odds)
        engine.make_moves_from_current_position([move])
        board.push(chess.Move.from_uci(move))
        fen = board.fen()
        print(move)
        print(fen)
    print()
    turn = fen.split(" ")[1]
    if turn == "w":
        print("White to move")
    elif turn == "b":
        print("Black to move")
    print()
    print(engine.get_board_visual())
    print()
    print("Evaluation:", val)
    print()
    del engine
    return fen


def opposite_color_bishops(row):
    idx = 0
    for i, piece in enumerate(row):
        if piece.lower() == "b":
            idx += i
    return idx % 2 == 1


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, default="./stockfish", help="path to Stockfish engine binary (default ./stockfish)")
    parser.add_argument("--depth", type=int, default=20, help="evaluation depth (default 20)")
    parser.add_argument("--threads", type=int, default=1, help="engine CPU threads (default 1)")
    parser.add_argument("--hash", type=int, default=1024, help="engine hash size (default 1024)")
    parser.add_argument("--seed", type=int, default=None, help="random seed (default random)")
    parser.add_argument("--odds", type=float, default=0.0, help="target evaluation (default 0.0)")
    parser.add_argument("--tolerance", type=float, default=0.1, help="imbalance tolerance (default 0.1)")
    parser.add_argument("--out", type=str, default=None, help="output FEN to specified file")
    args = parser.parse_args()
    return args


def header():
    print()
    print("Funky Chess")
    # print(f"Version {version}")
    print("By Santiago Benoit")
    print()


def main():
    header()
    args = get_args()
    print("Arguments:", args)
    engine = Stockfish(path=args.engine)
    print("Stockfish version:", engine.get_stockfish_major_version())
    del engine
    seed = args.seed
    if seed is None:
        seed = secrets.randbits(32)
    print("Random seed:", seed)
    np.random.seed(seed)
    if args.threads > 1:
        print("Warning: random seed consistency is not guaranteed with threads > 1.")
    print("\nGenerating position...\n")
    white = ["R", "N", "B", "Q", "K", "B", "N", "R"]
    black = ["r", "n", "b", "q", "k", "b", "n", "r"]
    shuffled = False
    while not shuffled or not opposite_color_bishops(white):
        np.random.shuffle(white)
        shuffled = True
    shuffled = False
    while not shuffled or not opposite_color_bishops(black):
        np.random.shuffle(black)
        shuffled = True
    fen = "".join(black) + "/pppppppp/8/8/8/8/PPPPPPPP/" + "".join(white) + " w - - 0 1"
    fen2 = balance(fen, args)
    if args.out is not None:
        with open(args.out, "w") as f:
            f.write(fen2)
        print("Wrote FEN to", args.out)
        print()


if __name__ == "__main__":
    main()
