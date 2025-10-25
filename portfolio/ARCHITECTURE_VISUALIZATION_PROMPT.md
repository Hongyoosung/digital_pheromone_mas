# System Architecture Visualization Prompt

Use this prompt with AI image generation tools (DALL-E, Midjourney, Stable Diffusion) or diagram generation AI to create professional visual aids for the 4D Digital Pheromone Multi-Agent System project.

---

## Diagram 1: System Architecture Overview

**Prompt:**

```
Create a professional system architecture diagram showing a distributed multi-agent system with the following components:

MAIN COMPONENTS (arranged in layers):

1. TOP LAYER - "Agent Layer"
   - Multiple circular agent nodes (10-15 nodes) distributed across space
   - Each agent has 4 colored inner segments representing: Behavior (blue), Emotion (red), Social (green), Context (yellow)
   - Arrows showing agent movement and interaction

2. MIDDLE LAYER - "Pheromone Field Layer"
   - Grid-based spatial field with heat map visualization
   - Multiple overlapping colored gradients representing different pheromone dimensions
   - Diffusion waves emanating from agent positions
   - Color legend: Behavior=Blue, Emotion=Red, Social=Green, Context=Yellow

3. PROCESSING LAYER - "Neural Network Layer"
   - Left side: "Temporal Diffusion Model" with learnable decay parameters
   - Right side: "Distributed Attention Router" with multi-head attention visualization
   - Neural network nodes connected with weighted edges
   - GPU acceleration indicators (lightning bolts)

4. BOTTOM LAYER - "C++ Backend Acceleration"
   - Boxes labeled: "SIMD Field Operations", "Spatial Index", "Message Codec"
   - Performance metrics: "26x faster", "21x faster", "12x faster"
   - CPU/GPU icons

CONNECTIONS:
- Bidirectional arrows between agents and pheromone field (deposit/sense)
- Data flow arrows from pheromone field to neural networks
- Processing arrows from neural networks back to agents
- Acceleration arrows from C++ backend to all layers

STYLE:
- Professional technical diagram style
- Clean, modern aesthetic with soft shadows
- Use color coding consistently
- Include small annotations for key features
- Isometric or 2.5D perspective preferred
- White or light gray background
```

---

## Diagram 2: 4D Pheromone Vector Structure

**Prompt:**

```
Create a detailed infographic showing the 4-dimensional pheromone vector structure:

CENTER: Large 3D hexagonal prism labeled "4D Pheromone Vector"

FOUR BRANCHES extending from center (one per dimension):

1. BEHAVIOR BRANCH (Blue)
   - Icon: Robot action symbols
   - Sub-components: "Move", "Gather", "Attack", "Evade"
   - Mathematical notation: Softmax normalization formula
   - Bar chart showing probability distribution
   - Data flow: Raw action scores → Normalized probabilities

2. EMOTION BRANCH (Red)
   - Icon: Heart with emotion symbols
   - Five emotion circles: Happy (yellow), Fear (purple), Anger (orange), Sad (blue), Excited (pink)
   - Value range: [-1, 1] with min-max normalization visualization
   - Wave pattern showing emotional state over time

3. SOCIAL BRANCH (Green)
   - Icon: Network of connected nodes
   - Graph visualization with agent relationships
   - Edge weights based on interaction frequency
   - Heatmap showing relationship matrix
   - Formula: count_ij / T (normalized by time window)

4. CONTEXT BRANCH (Yellow)
   - Icon: Map/environment symbols
   - Sub-components: Position (x,y), Resource density, Threat proximity
   - Mini map with agent location highlighted
   - Feature vector with min-max scaling indicators
   - Environmental state visualization

BOTTOM SECTION:
- Timeline showing temporal decay of vector over time
- Decay formula: v(t) = v(0) × α^t
- Graph showing exponential decay curve

STYLE:
- Modern infographic design
- Clear icons and visual metaphors
- Color-coded by dimension
- Mathematical formulas in clean typography
- Arrows showing data transformation flow
```

---

## Diagram 3: Distributed Attention Routing Network

**Prompt:**

```
Create a technical diagram illustrating the distributed attention routing mechanism:

MAIN STRUCTURE:

1. INPUT LAYER (Bottom)
   - Array of 10 agent representations (small circles)
   - Each agent emits a pheromone vector (shown as colored arrows)
   - Labels: Agent 0, Agent 1, ... Agent 9

2. QUERY-KEY-VALUE ENCODING (Lower-Middle)
   - Three parallel processing streams:
     * Query stream (purple): Input projection → Agent-specific queries
     * Key stream (blue): Input projection → All agent keys
     * Value stream (green): Input projection → All agent values
   - Linear transformation boxes with parameter matrices

3. MULTI-HEAD ATTENTION (Center)
   - 8 attention heads shown as parallel processing units
   - Each head labeled: Head 1, Head 2, ..., Head 8
   - Attention weight matrices visualized as heatmaps
   - Self-attention connections between all agents
   - Color intensity showing attention weights (white=high, dark=low)

4. ROUTING MLP (Upper-Middle)
   - Feedforward network with:
     * Input layer (64 dims)
     * Hidden layer (128 dims) with ReLU activation
     * Output layer (64 dims) with LayerNorm
   - Dropout layers indicated
   - Neuron activation visualization

5. OUTPUT LAYER (Top)
   - Final routed messages for each agent
   - Arrows showing message distribution back to agents
   - Communication efficiency metrics display

SIDE PANELS:

LEFT PANEL - "Communication Topology"
- Three topology types visualized:
  * Full: Complete graph
  * Ring: Circular connections
  * Random: Sparse random graph
- Attention masks shown as binary matrices

RIGHT PANEL - "Training Metrics"
- Line graphs showing:
  * Routing loss over time
  * Attention entropy
  * Communication efficiency
- Performance indicators

STYLE:
- Clean neural network diagram aesthetic
- Gradient flows and attention connections
- Heatmap visualizations for weight matrices
- Professional AI/ML publication style
- Annotations for key mathematical operations
```

---

## Diagram 4: Temporal Diffusion Process

**Prompt:**

```
Create a visualization showing the temporal diffusion process across multiple timesteps:

LAYOUT: Horizontal timeline with 5 time slices (T=0, T=1, T=2, T=3, T=4)

FOR EACH TIME SLICE:
- 2D grid map (50x50 cells)
- Pheromone field intensity shown as heatmap
- Agent positions marked as white circles with IDs
- Diffusion radius indicated with translucent circles
- Color gradient from high intensity (bright) to low intensity (dark)

TIME PROGRESSION:
- T=0: Single agent deposits strong pheromone (bright red spot)
- T=1: Pheromone spreads to adjacent cells (Gaussian diffusion pattern)
- T=2: Further spreading, intensity decreases (decay applied)
- T=3: Wider spread, lower intensity
- T=4: Maximum spread, minimal intensity at edges

TOP SECTION - "Learnable Decay Function"
- Graph showing decay factor adaptation
- Neural temporal encoder architecture
- Formula: v(t) = v(0) × learnable_decay^t × temporal_weight(t)
- Activation curve showing sigmoid-shaped temporal weighting

BOTTOM SECTION - "GPU Acceleration"
- Comparison visualization:
  * Left: CPU diffusion (sequential cell updates)
  * Right: GPU diffusion (parallel convolution)
- Speed comparison: "12x faster with CUDA"
- Convolution kernel visualization (Gaussian filter)

ANNOTATIONS:
- Decay rate values at each timestep
- Pheromone magnitude measurements
- Diffusion radius indicators
- Color scale legend

STYLE:
- Scientific visualization aesthetic
- Smooth color gradients for heatmaps
- Clear timeline progression
- Professional simulation visualization style
```

---

## Diagram 5: Performance Benchmarking Results

**Prompt:**

```
Create a comprehensive performance dashboard showing benchmark results:

LAYOUT: 2x2 grid of visualizations

TOP-LEFT: "C++ Backend Speedup"
- Horizontal bar chart comparing Python vs C++ performance
- Bars for: Field Decay (26x), Field Aggregation (21x), Spatial Diffusion (12x), Spatial Queries (2-30x)
- Each bar split into two colors: Python (gray), C++ (blue)
- Speedup multipliers labeled on bars

TOP-RIGHT: "Scalability Analysis"
- Line graph with multiple curves
- X-axis: Number of agents (10, 50, 100, 200, 500)
- Y-axis: Execution time (seconds)
- Three lines: Proposed Method (green), Centralized Attention (red), Rule-Based (orange)
- Confidence interval shading

BOTTOM-LEFT: "Information Transfer Efficiency"
- Grouped bar chart
- X-axis: Experiment configurations
- Y-axis: Shannon Entropy
- Four bar groups: Proposed, 2D Ablation, Rule-Based, Centralized
- Error bars showing standard deviation
- Significance stars (* p<0.05, ** p<0.01)

BOTTOM-RIGHT: "Learning Convergence"
- Dual-axis line graph
- X-axis: Training epochs (0-1000)
- Left Y-axis: Cumulative reward
- Right Y-axis: Policy loss
- Multiple colored lines for different methods
- Shaded areas showing convergence threshold

CENTER BOTTOM: "System Resource Utilization"
- Stacked area chart showing:
  * GPU memory usage (red)
  * GPU compute utilization (orange)
  * CPU usage (blue)
  * Network I/O (green)
- Timeline across experiment duration
- Peak utilization markers

STYLE:
- Professional data visualization aesthetic
- Consistent color scheme across charts
- Clear axis labels and legends
- Grid lines for easy reading
- Statistical annotations (means, std dev, p-values)
```

---

## Diagram 6: Agent Decision-Making Flow

**Prompt:**

```
Create a flowchart showing the agent decision-making process:

FLOWCHART STRUCTURE (top to bottom):

1. START: "Agent Activation" (rounded rectangle, green)

2. SENSE PHASE (blue section)
   ├─ "Read Local Pheromone Field"
   ├─ "Decode 4D Pheromone Vectors"
   └─ "Aggregate Multi-Source Signals"

3. ATTENTION PHASE (purple section)
   ├─ "Generate Query from Self-State"
   ├─ "Compute Attention over Neighbors"
   ├─ "Weight Information by Attention Scores"
   └─ "Route Messages via Attention Network"

4. DECISION PHASE (orange section)
   ├─ "Encode Current State + Pheromone Input"
   ├─ "Forward Pass through Policy Network"
   ├─ "Sample Action from Distribution"
   └─ Decision Diamond: "Valid Action?"
       ├─ YES → Continue
       └─ NO → Return to "Sample Action"

5. ACTION PHASE (red section)
   ├─ "Execute Selected Action"
   ├─ "Update Internal State (emotion, memory)"
   ├─ "Observe Environment Feedback"
   └─ "Compute Reward Signal"

6. DEPOSIT PHASE (green section)
   ├─ "Generate 4D Pheromone Vector"
   │   ├─ Behavior: Action distribution
   │   ├─ Emotion: Current affective state
   │   ├─ Social: Relationship updates
   │   └─ Context: Environmental features
   ├─ "Apply Normalization"
   └─ "Deposit to Spatial Field"

7. LEARN PHASE (yellow section)
   ├─ "Store Experience in Replay Buffer"
   ├─ Decision Diamond: "Training Step?"
       ├─ YES → "Update Policy Network" → "Update Attention Router"
       └─ NO → Skip
   └─ "Update Temporal Diffusion Model"

8. END: "Increment Timestep" (rounded rectangle, green) → Loop back to START

SIDE ANNOTATIONS:
- Timing information for each phase
- Data structures passed between phases
- GPU/CPU execution indicators

STYLE:
- Professional flowchart with standard symbols
- Color-coded phases for easy identification
- Clear arrows showing flow direction
- Annotations for key operations
- Modern corporate aesthetic
```

---

## Diagram 7: Experimental Setup & Comparison

**Prompt:**

```
Create a comparison visualization showing experimental setup and baseline methods:

LAYOUT: Three-column comparison

LEFT COLUMN: "Proposed Method - 4D Digital Pheromone"
- Agent icon with 4 colored layers (behavior, emotion, social, context)
- Network diagram showing distributed attention connections
- Pheromone field with multi-dimensional visualization
- Icons: Neural network, GPU, distributed nodes
- Labels: "Learnable Diffusion", "Multi-Head Attention", "4D Vectors"
- Performance badge: "Baseline +10-15%"

MIDDLE COLUMN: "Baseline 1 - Rule-Based Diffusion"
- Simple agent icon
- Traditional gradient field visualization (single color)
- Static diffusion rules (no learning)
- Icons: Fixed rules, gradient arrows
- Labels: "Static Decay", "Gradient Following", "1D Signal"
- Performance marker: "Reference"

RIGHT COLUMN: "Baseline 2 - Centralized Attention"
- Agent icons connected to central server
- Star topology network diagram
- Server icon with high load indicator
- Icons: Central hub, bottleneck warning
- Labels: "Server Routing", "Single Point", "Full Information"
- Performance marker: "Scalability Issues"

BOTTOM SECTION: "Experimental Variables"
- Table showing configuration matrix:
  * Agents: 10, 50, 500
  * Timesteps: 100, 1000, 5000
  * Map size: 25x25, 50x50
  * GPU count: 1-4
- Checkmarks indicating tested combinations

TOP SECTION: "Evaluation Metrics"
- Icons with metric names:
  * Gauge icon: "Shannon Entropy"
  * Upward arrow: "Convergence Speed"
  * Network icon: "Communication Overhead"
  * Server icon: "Network Load"
- Formula boxes for each metric

STYLE:
- Clean comparison layout
- Consistent visual language across columns
- Icons and symbols for quick comprehension
- Professional academic presentation style
```

---

## Alternative Prompt for Unified Architecture Diagram

**Comprehensive System Prompt:**

```
Create a single comprehensive technical architecture diagram for a distributed multi-agent AI system with the following specifications:

OVERALL LAYOUT: Layered architecture with data flow from bottom to top

LAYER 1 (Foundation): C++ Acceleration Backend
- Dark blue background
- Three modules: SIMD Field Operations, Spatial Index, Message Codec
- Performance metrics displayed: 26x, 21x, 12x speedup indicators
- CPU/GPU icons

LAYER 2: Pheromone Field Management
- 2D grid visualization (50x50 cells)
- Multi-colored heatmap overlays (4 colors representing 4 dimensions)
- Spatial diffusion visualization with Gaussian kernels
- Temporal decay animations indicated by fading gradients

LAYER 3: Neural Processing
Left Half: Temporal Diffusion Model
- Learnable decay parameter block
- Temporal encoder network diagram
- Vector projection layers
Right Half: Distributed Attention Router
- Multi-head attention visualization (8 heads)
- Q, K, V projection blocks
- Routing MLP network
- Output aggregation

LAYER 4: Agent Layer
- 10-15 agent nodes distributed across space
- Each agent shows 4-part internal structure (4D vector components)
- Attention connections between agents (curved lines with weights)
- Movement trajectories shown as dotted paths

LAYER 5 (Top): Metrics & Monitoring
- Real-time performance dashboards
- Learning curves graph
- Communication efficiency metrics
- System resource utilization bars

DATA FLOW INDICATORS:
- Upward arrows: Agent observations → Pheromone sensing
- Horizontal arrows: Inter-agent attention-based communication
- Downward arrows: Pheromone deposition
- Bidirectional arrows: Learning signal backpropagation

ANNOTATIONS:
- Key equations at relevant components
- Dimension labels for tensors
- Timing information for operations
- Color legend for pheromone dimensions

VISUAL STYLE:
- Professional technical documentation quality
- Consistent color scheme (blue=behavior, red=emotion, green=social, yellow=context)
- Clean lines and modern aesthetic
- Sufficient whitespace for readability
- Typography: Sans-serif, clear hierarchy
- Perspective: Slight isometric view for depth
```

---

## Usage Instructions

1. **For AI Image Generators (DALL-E, Midjourney, Stable Diffusion):**
   - Use the prompts as-is, possibly splitting complex diagrams into multiple images
   - Adjust style modifiers based on the tool's capabilities
   - May need to simplify technical details for artistic generators

2. **For Diagram Generation AI (Mermaid AI, Lucidchart AI, Excalidraw AI):**
   - Use the structured descriptions to guide node/edge creation
   - Focus on the component hierarchy and connection descriptions
   - Leverage the layout specifications

3. **For Manual Creation:**
   - Use these prompts as detailed specifications
   - Follow the color coding and layout guidelines
   - Ensure consistency across all diagrams

4. **Customization:**
   - Adjust complexity based on audience (technical vs. general)
   - Modify color schemes to match presentation templates
   - Scale diagram elements based on output medium (poster, paper, slides)

---

## Recommended Tools

- **Mermaid.js**: For flowcharts and sequence diagrams
- **PlantUML**: For system architecture diagrams
- **TikZ/LaTeX**: For publication-quality technical diagrams
- **Draw.io/Diagrams.net**: For manual creation with AI assistance
- **Python (Matplotlib/Seaborn)**: For performance visualization and heatmaps
- **D3.js**: For interactive web-based visualizations

---

## Color Palette Reference

**Primary Dimensions:**
- Behavior: #3498DB (Blue)
- Emotion: #E74C3C (Red)
- Social: #2ECC71 (Green)
- Context: #F39C12 (Yellow/Orange)

**System Components:**
- Neural Networks: #9B59B6 (Purple)
- C++ Backend: #34495E (Dark Gray)
- GPU Acceleration: #1ABC9C (Teal)
- Agents: #ECF0F1 (Light Gray with colored segments)

**Performance Indicators:**
- Positive/Improvement: #27AE60 (Green)
- Warning: #F39C12 (Orange)
- Error/Bottleneck: #E74C3C (Red)
- Neutral: #95A5A6 (Gray)
