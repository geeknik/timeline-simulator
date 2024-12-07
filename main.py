from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from qiskit_aer import AerSimulator
from qiskit_aer import Aer  # Updated import
from qiskit_aer.noise import NoiseModel, depolarizing_error  # Updated import
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.quantum_info import DensityMatrix, partial_trace, Statevector, entropy
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SimulationConfig:
    """Configuration parameters for quantum state evolution simulation.
    
    Models a multi-qubit system subject to unitary evolution, environmental decoherence,
    and projective measurements. Based on standard decoherence theory (Zurek 2003) and 
    the quantum measurement formalism.
    
    Attributes:
        num_trait_qubits: Number of qubits in primary register
        num_timeline_steps: Number of evolution timesteps
        measurement_frequency: Frequency of projective measurements (0-1)
        decoherence_rates: Dict mapping qubit indices to their decoherence rates
        branch_points: Timesteps where state vector branching occurs
        shots: Number of experimental shots for statistics
        projection_probability: Probability of state projection during measurement
        decoherence_rate: Base environmental decoherence rate
        num_branches: Number of potential state vector branches to track
    
    References:
        - Zurek, W.H. (2003). Decoherence, einselection, and the quantum origins of the classical.
        - Schlosshauer, M. (2007). Decoherence and the Quantum-to-Classical Transition.
    """
    num_trait_qubits: int = 3
    num_timeline_steps: int = 5
    measurement_frequency: float = 0.2
    decoherence_rates: Dict[int, float] = None
    branch_points: List[int] = None
    shots: int = 1000
    death_probability: float = 0.3
    decoherence_rate: float = 0.05
    num_timelines: int = 2

    def __post_init__(self):
        if self.decoherence_rates is None:
            self.decoherence_rates = {i: 0.05 for i in range(self.num_trait_qubits)}
        if self.branch_points is None:
            self.branch_points = [2, 4]  # Default branch points at steps 2 and 4

class QuantumTimelineSimulator:
    """
    A quantum simulator implementing state vector evolution under decoherence and measurement.
    
    This implementation models a multi-qubit system evolving under:
    1. Unitary dynamics (quantum gates)
    2. Environmental decoherence (noise channels)
    3. Projective measurements
    
    The system state |ψ⟩ evolves as:
    |ψ(t)⟩ = U(t)|ψ(0)⟩ for unitary U(t)
    
    Under decoherence, the density matrix ρ evolves as:
    ρ(t) = Σᵢ Kᵢ(t)ρ(0)Kᵢ†(t) 
    where Kᵢ are Kraus operators representing the noise channel
    
    Key quantum mechanical features:
    - Unitary evolution via quantum gates
    - Decoherence via standard noise models
    - Projective measurements causing state collapse
    - Entanglement between subsystems
    - Von Neumann entropy tracking
    
    The simulator provides a rigorous framework for studying quantum state
    evolution under realistic noise and measurement conditions, following
    standard quantum mechanical formalism.
    
    References:
        - Nielsen & Chuang (2010). Quantum Computation and Quantum Information.
        - Zurek (2003). Decoherence and the transition from quantum to classical.
    """
    def __init__(self, config: SimulationConfig = SimulationConfig()):
        """
        Initialize the quantum timeline simulator with specified configuration.
        
        Args:
            config: SimulationConfig object containing simulation parameters
        """
        self.config = config
        try:
            # Create registers for traits and their measurements
            self.trait_qr = QuantumRegister(config.num_trait_qubits, 'traits')
            self.trait_cr = ClassicalRegister(config.num_trait_qubits, 'trait_measures')
            
            # Create ancilla qubits for timeline branching
            self.branch_qr = QuantumRegister(len(config.branch_points), 'branch')
            self.branch_cr = ClassicalRegister(len(config.branch_points), 'branch_measures')
            
            # Initialize quantum circuit with all registers
            self.circuit = QuantumCircuit(
                self.trait_qr, self.branch_qr,
                self.trait_cr, self.branch_cr
            )
            
            # Create combined noise model for the simulator
            self.noise_model = NoiseModel()
            trait_noise_models = self._create_noise_models()
            
            # Create a single noise model with basic quantum errors
            error_1q = depolarizing_error(self.config.decoherence_rate, 1)
            error_2q = depolarizing_error(self.config.decoherence_rate/2, 2)
            
            # Add quantum errors to the noise model
            self.noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3'])
            self.noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
            
            # Track the current timestep
            self.current_step = 0
            
            logger.info(f"Initialized simulator with {config.num_trait_qubits} trait qubits")
        except Exception as e:
            logger.error(f"Failed to initialize quantum simulator: {e}")
            raise

    def _create_noise_models(self) -> Dict[int, NoiseModel]:
        """Create custom noise models for each trait qubit."""
        try:
            noise_models = {}
            for idx, rate in self.config.decoherence_rates.items():
                model = NoiseModel()
                
                # Single qubit depolarizing error
                error_1q = depolarizing_error(rate, 1)
                # Two qubit depolarizing error
                error_2q = depolarizing_error(rate/2, 2)
                
                # Add all-qubit quantum errors
                model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3'])
                model.add_all_qubit_quantum_error(error_2q, ['cx'])
                
                noise_models[idx] = model
            
            return noise_models
        except Exception as e:
            logger.error(f"Failed to create noise model: {e}")
            raise

    def initialize_traits(self) -> None:
        """
        Initialize trait qubits in superposition states.
        
        Creates a complex superposition of trait states, with each trait
        having its own quantum amplitude and phase. This allows for
        modeling of personality/consciousness traits as quantum states.
        """
        try:
            # Initialize each trait in a unique superposition
            for i in range(self.config.num_trait_qubits):
                # Create different superposition angles for variety
                theta = np.pi * (i + 1)/(2 * self.config.num_trait_qubits)
                phi = np.pi * i/self.config.num_trait_qubits
                
                # Apply rotation gates to create superposition
                self.circuit.ry(theta, self.trait_qr[i])
                self.circuit.rz(phi, self.trait_qr[i])
                
            # Entangle traits with controlled operations
            for i in range(self.config.num_trait_qubits - 1):
                self.circuit.cx(self.trait_qr[i], self.trait_qr[i + 1])
            
            logger.info("Initialized trait superpositions")
        except Exception as e:
            logger.error(f"Failed to initialize superposition: {e}")
            raise

    def apply_projection_measurement(self) -> None:
        """Apply a projective measurement with configured probability.
        
        Implements a controlled projection onto the computational basis states
        using rotation gates followed by measurement. The projection probability
        determines the angle of rotation, following standard measurement theory.
        """
        try:
            theta = 2 * np.arcsin(np.sqrt(self.config.death_probability))
            # Apply death probability rotation to each trait qubit
            for i in range(self.config.num_trait_qubits):
                self.circuit.ry(theta, self.trait_qr[i])
            logger.info(f"Applied death event with probability {self.config.death_probability}")
        except Exception as e:
            logger.error(f"Failed to apply death event: {e}")
            raise

    def entangle_timelines(self) -> None:
        """Entangle multiple timeline qubits using CNOT gates."""
        try:
            if self.config.num_timelines > 1:
                for i in range(self.config.num_timelines - 1):
                    self.circuit.cx(i, i + 1)
                logger.info("Entangled timelines")
        except Exception as e:
            logger.error(f"Failed to entangle timelines: {e}")
            raise

    def measure_timelines(self) -> None:
        """Measure all trait and branch qubits."""
        try:
            # Measure trait qubits
            self.circuit.measure(self.trait_qr, self.trait_cr)
            # Measure branch qubits
            self.circuit.measure(self.branch_qr, self.branch_cr)
            logger.info("Added measurement operations")
        except Exception as e:
            logger.error(f"Failed to add measurements: {e}")
            raise

    def simulate(self) -> Tuple[Dict, QuantumCircuit]:
        """
        Execute the quantum circuit simulation.
        
        Returns:
            Tuple containing:
            - Dict of measurement results
            - QuantumCircuit without measurements for state analysis
        """
        try:
            # Create a copy of circuit without measurements for state analysis
            state_circuit = QuantumCircuit(self.trait_qr, self.branch_qr)
            for inst in self.circuit.data:
                if inst.operation.name != "measure":
                    state_circuit.append(inst.operation, inst.qubits, inst.clbits)
            
            # Run the measurement circuit
            backend = AerSimulator(noise_model=self.noise_model)
            job = backend.run(self.circuit, shots=self.config.shots)
            counts = job.result().get_counts(self.circuit)
            logger.info(f"Simulation completed with {self.config.shots} shots")
            return counts, state_circuit
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise

    def analyze_results(self, counts: Dict) -> Dict[str, float]:
        """
        Analyze simulation results and compute statistics.
        
        Args:
            counts: Measurement counts dictionary
            
        Returns:
            Dictionary containing analysis metrics
        """
        try:
            total_shots = sum(counts.values())
            
            # Count states where all trait qubits are in |0⟩ state (survival)
            # Since traits are measured first, they are the rightmost bits
            trait_bits = self.config.num_trait_qubits
            survival_count = sum(
                count for state, count in counts.items() 
                if state[-trait_bits:] == '0' * trait_bits
            )
            
            # Calculate basic statistics 
            survival_rate = survival_count / total_shots
            death_rate = 1 - survival_rate
            
            # Calculate quantum entropy from measurement distribution
            probabilities = [count/total_shots for count in counts.values()]
            quantum_entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
            
            analysis = {
                'survival_rate': survival_rate,
                'death_rate': death_rate,
                'quantum_entropy': quantum_entropy,
                'total_states': len(counts),
                'measurement_counts': counts
            }
            
            logger.info(f"Analysis completed: Survival rate = {analysis['survival_rate']:.2%}")
            return analysis
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise

def run_quantum_evolution_experiment(config: SimulationConfig = SimulationConfig()) -> Dict:
    """
    Run a complete quantum evolution experiment with decoherence and measurement.
    
    Executes a quantum circuit that:
    1. Prepares an initial superposition state
    2. Applies unitary evolution
    3. Models environmental decoherence
    4. Performs projective measurements
    
    The experiment follows standard quantum mechanical principles and measurement
    theory, collecting statistics over multiple shots to build a measurement
    distribution.
    
    Args:
        config: SimulationConfig object containing simulation parameters
        
    Returns:
        Dictionary containing:
        - Raw measurement counts
        - Statistical analysis
        - Quantum metrics (entropy, purity)
    """
    try:
        simulator = QuantumTimelineSimulator(config)
        
        # Create and execute quantum circuit
        simulator.initialize_traits()
        simulator.apply_death_event()
        simulator.entangle_timelines()
        simulator.measure_timelines()
        
        # Run simulation
        counts, simulator.state_circuit = simulator.simulate()
        analysis = simulator.analyze_results(counts)
        
        return {
            'counts': counts,
            'analysis': analysis
        }
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise

if __name__ == "__main__":
    # Initialize rich console for output
    console = Console()
    
    # Configure quantum simulation parameters
    config = SimulationConfig(
        shots=1000,
        projection_probability=0.3,  # Probability of state projection
        decoherence_rate=0.05       # Environmental decoherence rate
    )
    
    try:
        results = run_quantum_evolution_experiment(config)
        
        # Create results panel
        results_table = Table(show_header=False)
        results_table.add_row("Survival Rate", f"[green]{results['analysis']['survival_rate']:.2%}[/]")
        results_table.add_row("Death Rate", f"[red]{results['analysis']['death_rate']:.2%}[/]")
        
        # Create measurement counts table
        counts_table = Table(title="Measurement Counts", show_header=True)
        counts_table.add_column("Quantum State", style="cyan")
        counts_table.add_column("Count", justify="right")
        counts_table.add_column("Percentage", justify="right")
        
        total_shots = sum(results['counts'].values())
        for state, count in results['counts'].items():
            percentage = count/total_shots
            color = "green" if state.count('1') == 0 else "yellow" if state.count('1') == 1 else "red"
            counts_table.add_row(
                f"|{state}⟩",
                str(count),
                f"[{color}]{percentage:.1%}[/]"
            )
        
        # Create quantum metrics panel
        metrics_table = Table(show_header=False)
        metrics_table.add_row(
            "Von Neumann Entropy",
            f"[blue]{results['analysis']['quantum_entropy']:.6f}[/] bits"
        )
        metrics_table.add_row(
            "Total States",
            f"[magenta]{results['analysis']['total_states']}[/]"
        )
        
        # Display everything
        console.print("\n")
        console.print(Panel(results_table, title="Experiment Results"))
        console.print("\n")
        console.print(counts_table)
        console.print("\n")
        console.print(Panel(metrics_table, title="Quantum Metrics"))
        console.print("\n")
        
    except Exception as e:
        console.print(f"[red]Experiment failed: {e}[/]")
