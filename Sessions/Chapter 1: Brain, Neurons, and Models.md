# Artificial Neural Networks - Chapter 1: Brain, Neurons, and Models
---

### **1. Introduction to Brain and Neurons**

- **Human Brain and Biological Neurons**
The human brain contains over 100 billion neurons, each forming thousands of connections with other neurons through synapses. These connections form neural networks, totaling over 1,000 trillion synapses.
Key facts:
    - Neurons transmit electrical potential over axons and dendrites.
    - A neuron fires when the signal exceeds a threshold and passes it along to other neurons.
    - Synapses are key to signal transmission, with neuromuscular junctions connecting neurons to muscle cells.

### **2. How Neurons Work**

- Neurons work by switching to an excited state when they receive sufficient signals. They then transmit electrical signals via synapses to other neurons. The transmission of these signals is mediated by neurotransmitters, which are critical for memory formation.
    - Synaptic learning: successful transmissions strengthen synaptic connections, while failed ones weaken them.
    - Memory formation involves the strengthening of synaptic connections, a process known as long-term potentiation.

### **3. McCulloch & Pitts Neuron Model**

- **Warren McCulloch and Walter Pitts**: Developed the first computational model of a neuron in 1943, known as the McCulloch and Pitts neuron.
    - This model simplifies the behavior of natural neurons by treating inputs as Boolean (0 or 1) and firing if the aggregated input exceeds a threshold.
    - Unlike natural neurons, the McCulloch and Pitts model does not account for time-based accumulation of signals or refractory periods. It also omits inhibiting inputs, which are critical in natural neurons.

### **4. Properties of McCulloch and Pitts Neuron**

- **Logic Operations**: The McCulloch and Pitts neuron can simulate logical functions such as AND, OR, and NOT by adjusting input signals and thresholds. More complex operations like XOR require combining multiple neurons.
- **Learning Rule**: Neurons can learn by adjusting weights on inputs. For example, by modifying the weight vector, the neuron can classify points differently, performing functions like OR or AND.

### **5. Artificial Neuron General Properties**

- **Activation Function**: Artificial neurons use an activation function, typically a step-function, to determine whether they should fire.
    - The input signal is weighted and aggregated. If the aggregated signal exceeds a threshold, the neuron fires.
    - Unlike natural neurons, artificial neurons lack a refractory period and fire as long as the input signal exceeds the threshold.

### **6. Improvements for Artificial Neurons**

- Future developments in artificial neural networks should aim to include more natural neuron-like behaviors, such as:
    - Spike activation functions with refractory periods.
    - Accumulation of signals over time.
    - Synaptic properties, including inhibiting synapses.
    - Improved network connectivity.

### **7. Conclusion**

- The McCulloch and Pitts neuron is the foundation for modern artificial neural networks, but further improvements are needed to bring artificial neurons closer to their natural counterparts.