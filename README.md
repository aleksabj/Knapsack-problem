# Knapsack Problem - Evolutionary Algorithm

## Description
This project implements an evolutionary algorithm to solve the 0/1 knapsack problem. The algorithm uses a genetic approach to evolve a population of solutions over multiple generations, optimizing for the highest total value while respecting the weight constraint.

The code contains comments explaining what I have done at each step.

## Algorithm Details
### Encoding of Individuals
Each individual in the population is represented as a binary string, where each bit corresponds to an item in the knapsack: (`def initialize_population()`)
- `1` means the item is included in the knapsack.
- `0` means the item is not included.

### Genetic Operators
- **Selection**: Tournament selection is used to choose parents for the next generation. (`def tournament_selection()`)
- **Crossover**: A one-point crossover method is applied to combine genetic material from two parents. (`def crossover()`)
- **Mutation**: A mutation operator randomly flips bits in the binary string with a small probability of introducing variation. (`def mutate()`)

### Fitness Function
The fitness of an individual is calculated as the total value of selected items while ensuring the total weight does not exceed the knapsack’s capacity. (`def evaluate()`)

## Running the Code
To run the algorithm, ensure you have Python installed. Install dependencies with:
```sh
pip install -r requirements.txt
```
Then, execute the script with:
```sh
python src/main.py data/<input_file>
```
where `<input_file>` is one of the provided dataset files (e.g., `debug_10.txt`, `input_100.txt`).

## Best Solutions Discovered
The best solutions I obtained for each dataset are:

| Input File        | Best Solution Value |
|------------------|------------------|
| `debug_10.txt`  | 295              |
| `debug_20.txt`  | 1024             |
| `input_100.txt` | 9147             |
| `input_1000.txt`| 52513            |

For `debug_10.txt` and `debug_20.txt`, the algorithm successfully found the optimal solutions, confirming its correctness.

## Project Structure
```
Knapsack-problem/
│-- data/       # Contains input files (debug_10.txt, debug_20.txt, input_100.txt, input_1000.txt)
│-- results/    # Stores generated graphs
│-- src/        # Contains main.py
│-- requirements.txt
│-- README.md
```

## Results Visualization
Graphs tracking the fitness evolution across generations are saved in the `results/` directory.
