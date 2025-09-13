# Quantum Timeline Simulator

A sophisticated quantum mechanical simulator that models multi-qubit systems evolving under unitary dynamics, environmental decoherence, and measurement operations.

## Overview

This project implements a quantum simulator that models complex quantum systems using rigorous quantum mechanical principles, including:

- Unitary evolution via quantum gates
- Environmental decoherence using standard noise models
- Projective measurements and state collapse
- Quantum entanglement between subsystems
- Von Neumann entropy tracking

The simulator is built on Qiskit and provides a framework for studying quantum state evolution under realistic conditions.

## Features

- Multi-qubit system simulation
- Customizable decoherence rates
- Configurable measurement frequencies
- Timeline branching and entanglement
- Rich visualization of results
- Comprehensive quantum metrics

## Requirements

- Python 3.8+

```python
qiskit
qiskit-aer
numpy
matplotlib
rich
```

## Installation

```bash
pip install qiskit qiskit-aer numpy matplotlib rich
```

## Usage

Basic usage:

```python
from main import SimulationConfig, run_quantum_evolution_experiment

# Configure simulation parameters
config = SimulationConfig(
    shots=1000,
    death_probability=0.3,
    decoherence_rate=0.05
)

# Run experiment
results = run_quantum_evolution_experiment(config)
```

Run the main script directly:

```bash
python main.py
```

## Output

The simulator provides detailed output including:
- Survival/death rates
- Measurement counts for each quantum state
- Von Neumann entropy
- Total unique states observed
- Rich console visualization of results

## Scientific Background

The implementation is based on standard quantum mechanical formalism and decoherence theory. Key references:

- Nielsen, M. A., & Chuang, I. L. (2010). Quantum Computation and Quantum Information: 10th Anniversary Edition. Cambridge: Cambridge University Press.
- Zurek (2003). Decoherence and the transition from quantum to classical
- Schlosshauer (2007). Decoherence and the Quantum-to-Classical Transition

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
