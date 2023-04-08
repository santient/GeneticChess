# Genetic Chess
Balanced, randomized, and asymmetrical chess position generation using a genetic algorithm with Stockfish evaluation.

```

                                 |\_
O       o O       o O       o   /  .\_
| O   o | | O   o | | O   o |  |   ___)
| | O | | | | O | | | | O | |  |    \
| o   O | | o   O | | o   O |  |  =  |
o       O o       O o       O  /_____\
                              [_______]

Genetic Chess
By Santiago Benoit

Arguments: Namespace(stockfish='./stockfish', depth=15, seed=None, generations=15, population_size=10, mutation_rate=0.03125, final_depth=25)
Random seed: 3752869713

[Generation 0] [Evaluation -12.01] [FEN 8/2r3n1/p1NpQNp1/1nPrBK2/1np1P3/1P1Prk2/1P2p3/8 w - - 0 1]
[Generation 1] [Evaluation -8.47] [FEN 8/6n1/p1NpQNp1/1nPrBK2/1np1P3/1P1Prk2/1P2p3/8 w - - 0 1]
[Generation 2] [Evaluation -7.7] [FEN 8/6n1/p1NpQNp1/1nPrBK1P/1np1P3/1P1Prk2/1P2p3/8 w - - 0 1]
[Generation 3] [Evaluation -4.81] [FEN 8/6n1/p1NpQNp1/1nPrBK1P/1np1P3/1P1P1k2/1P2p3/8 w - - 0 1]
[Generation 4] [Evaluation 0.69] [FEN 8/6n1/p1N1QNp1/1nPrBK1P/2p1P3/1P1P1k2/1P2p3/8 w - - 0 1]
[Generation 5] [Evaluation 0.69] [FEN 8/6n1/p1N1QNp1/1nPrBK1P/2p1P3/1P1P1k2/1P2p3/8 w - - 0 1]
[Generation 6] [Evaluation 0.69] [FEN 8/6n1/p1N1QNp1/1nPrBK1P/2p1P3/1P1P1k2/1P2p3/8 w - - 0 1]
[Generation 7] [Evaluation 0.69] [FEN 8/6n1/p1N1QNp1/1nPrBK1P/2p1P3/1P1P1k2/1P2p3/8 w - - 0 1]
[Generation 8] [Evaluation 0.65] [FEN 8/6n1/p1N1QNp1/1nPrBK1P/n1p1P3/1P1P1k2/1P2p3/8 w - - 0 1]
[Generation 9] [Evaluation 0.65] [FEN 8/6n1/p1N1QNp1/1nPrBK1P/n1p1P3/1P1P1k2/1P2p3/8 w - - 0 1]
[Generation 10] [Evaluation -0.35] [FEN 8/6n1/p1N1QNp1/pnPrBK1P/n1p1P3/1P1P1k2/1P2p3/8 w - - 0 1]
[Generation 11] [Evaluation 0.24] [FEN 8/6n1/p1N1QNp1/pnPrBK1P/p1p1P3/1P1P1k2/1P2p3/8 w - - 0 1]
[Generation 12] [Evaluation 0.24] [FEN 8/6n1/p1N1QNp1/pnPrBK1P/p1p1P3/1P1P1k2/1P2p3/8 w - - 0 1]
[Generation 13] [Evaluation 0.03] [FEN 3n4/Q5n1/p1N1QNq1/pnPrBK1P/p1p1P3/1P1P1k2/1P2p3/8 w - - 0 1]
[Generation 14] [Evaluation 0.03] [FEN 3n4/Q5n1/p1N1QNq1/pnPrBK1P/p1p1P3/1P1P1k2/1P2p3/8 w - - 0 1]
[Generation 15] [Evaluation 0.03] [FEN 3n4/Q5n1/p1N1QNq1/pnPrBK1P/p1p1P3/1P1P1k2/1P2p3/8 w - - 0 1]

RESULTS

+---+---+---+---+---+---+---+---+
|   |   |   | n |   |   |   |   | 8
+---+---+---+---+---+---+---+---+
| Q |   |   |   |   |   | n |   | 7
+---+---+---+---+---+---+---+---+
| p |   | N |   | Q | N | q |   | 6
+---+---+---+---+---+---+---+---+
| p | n | P | r | B | K |   | P | 5
+---+---+---+---+---+---+---+---+
| p |   | p |   | P |   |   |   | 4
+---+---+---+---+---+---+---+---+
|   | P |   | P |   | k |   |   | 3
+---+---+---+---+---+---+---+---+
|   | P |   |   | p |   |   |   | 2
+---+---+---+---+---+---+---+---+
|   |   |   |   |   |   |   |   | 1
+---+---+---+---+---+---+---+---+
  a   b   c   d   e   f   g   h

Evaluation: 0.0
FEN: 3n4/Q5n1/p1N1QNq1/pnPrBK1P/p1p1P3/1P1P1k2/1P2p3/8 w - - 0 1

```
