# Genetic Chess
## Balanced, asymmetrical chess position generation using genetic algorithms and Stockfish evaluation.

![Generated Position 1](position1.png)
2nrk1br/pppbb1p1/3rp1np/3r1p2/7P/PPP1P3/2QPNPP1/1R1KBQ1B w - - 0 1

![Generated Position 2](position2.png)
![Generated Position 3](position3.png)

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

Arguments: Namespace(stockfish='./stockfish', depth=20, final_depth=30, seed=None, odds=0.0, target_error=0.1)
Stockfish version: 15
Random seed: 2252179159

Evolving structure...

Evolving balance...

[Generation 0] [Evaluation 3.47] [FEN 2nrk1br/ppppb1p1/3rp1np/3r1p2/7P/PPP1P3/2QPNPP1/NR1KBQ2 w - - 0 1]
[Generation 1] [Evaluation 0.48] [FEN 2nrk1br/ppppb1p1/3rp1np/3r1p2/7P/PPP1P3/2QPNPP1/1R1KBQ2 w - - 0 1]
[Generation 2] [Evaluation 0.48] [FEN 2nrk1br/ppppb1p1/3rp1np/3r1p2/7P/PPP1P3/2QPNPP1/1R1KBQ2 w - - 0 1]
[Generation 3] [Evaluation -0.02] [FEN 2nrk1br/pppbb1p1/3rp1np/3r1p2/7P/PPP1P3/2QPNPP1/1R1KBQ1B w - - 0 1]

Final analysis...

RESULTS

+---+---+---+---+---+---+---+---+
|   |   | n | r | k |   | b | r | 8
+---+---+---+---+---+---+---+---+
| p | p | p | b | b |   | p |   | 7
+---+---+---+---+---+---+---+---+
|   |   |   | r | p |   | n | p | 6
+---+---+---+---+---+---+---+---+
|   |   |   | r |   | p |   |   | 5
+---+---+---+---+---+---+---+---+
|   |   |   |   |   |   |   | P | 4
+---+---+---+---+---+---+---+---+
| P | P | P |   | P |   |   |   | 3
+---+---+---+---+---+---+---+---+
|   |   | Q | P | N | P | P |   | 2
+---+---+---+---+---+---+---+---+
|   | R |   | K | B | Q |   | B | 1
+---+---+---+---+---+---+---+---+
  a   b   c   d   e   f   g   h

Evaluation: -1.08
FEN: 2nrk1br/pppbb1p1/3rp1np/3r1p2/7P/PPP1P3/2QPNPP1/1R1KBQ1B w - - 0 1
```

## Installation
1. Clone this repository
2. `pip install -r requirements.txt`
3. Download the latest Stockfish binary for your system: https://stockfishchess.org/download/

## Usage
```
python GeneticChess.py [-h] [--stockfish STOCKFISH] [--depth DEPTH] [--final-depth FINAL_DEPTH]
                       [--seed SEED] [--odds ODDS] [--target-error TARGET_ERROR]

options:
  -h, --help            show this help message and exit
  --stockfish STOCKFISH
                        path to stockfish binary
  --depth DEPTH
  --final-depth FINAL_DEPTH
  --seed SEED
  --odds ODDS
  --target-error TARGET_ERROR
```
