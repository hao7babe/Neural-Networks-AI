class McCullochPittsNeuron:
    def __init__(self, num_inputs, threshold):
        """
        Initializes the neuron with a specific number of inputs and a threshold.
        :param num_inputs: Number of inputs to the neuron
        :param threshold: Firing threshold for the neuron
        """
        self.num_inputs = num_inputs
        self.threshold = threshold
        self.weights = [1] * num_inputs  # Initialize all weights to 1

    def set_weights(self, weights):
        """
        Set the weights for the inputs.
        :param weights: A list of weights to apply to each input.
        """
        if len(weights) != self.num_inputs:
            raise ValueError("Number of weights must match the number of inputs.")
        self.weights = weights

    def activate(self, inputs):
        """
        Activate the neuron based on the input values.
        :param inputs: A list of binary inputs (0 or 1).
        :return: Output of the neuron (0 or 1).
        """
        if len(inputs) != self.num_inputs:
            raise ValueError("Number of inputs must match the number of neuron inputs.")

        # Compute the weighted sum of inputs
        weighted_sum = sum(w * i for w, i in zip(self.weights, inputs))

        # Apply threshold to determine if the neuron should fire
        return 1 if weighted_sum >= self.threshold else 0

    def __repr__(self):
        return f"McCullochPittsNeuron(inputs={self.num_inputs}, threshold={self.threshold}, weights={self.weights})"


# Enhanced user interaction
if __name__ == "__main__":
    print("Welcome to the McCulloch-Pitts Neuron Simulator!")
    print("This program simulates a simple neuron based on your inputs.")
    print("You will be prompted to enter the number of inputs, threshold value, and the input signals.")
    print("Let's get started!\n")

    # Step 1: Get the number of inputs
    while True:
        try:
            num_inputs = int(input("1. Please enter the number of inputs for the neuron (e.g., 3): "))
            if num_inputs <= 0:
                print("   The number of inputs must be a positive integer. Please try again.")
                continue
            break
        except ValueError:
            print("   Invalid input. Please enter a positive integer.")

    # Step 2: Get the threshold value
    while True:
        try:
            threshold = float(input("2. Please enter the threshold value for the neuron (e.g., 2): "))
            break
        except ValueError:
            print("   Invalid input. Please enter a numerical value for the threshold.")

    # Create the neuron
    neuron = McCullochPittsNeuron(num_inputs=num_inputs, threshold=threshold)

    # Optionally, set custom weights
    print("\nWould you like to set custom weights for each input?")
    custom_weights_choice = input("   Enter 'yes' to set custom weights or press Enter to use default weights of 1: ").strip().lower()

    if custom_weights_choice == 'yes':
        weights = []
        print("   Please enter the weight for each input:")
        for i in range(num_inputs):
            while True:
                try:
                    weight = float(input(f"     Weight for input {i+1}: "))
                    weights.append(weight)
                    break
                except ValueError:
                    print("       Invalid input. Please enter a numerical value for the weight.")
        neuron.set_weights(weights)
    else:
        print("   Using default weights of 1 for all inputs.")

    # Step 3: Get the input signals
    print("\n3. Now, please enter the binary input signals (0 or 1) for each input:")
    inputs = []
    for i in range(num_inputs):
        while True:
            try:
                inp = int(input(f"   Input signal {i+1} (0 or 1): "))
                if inp not in [0, 1]:
                    print("     Input must be 0 or 1. Please try again.")
                    continue
                inputs.append(inp)
                break
            except ValueError:
                print("     Invalid input. Please enter 0 or 1.")

    # Step 4: Activate the neuron and display the output
    output = neuron.activate(inputs)
    print("\nProcessing your inputs...")
    print(f"Neuron Output: {output}")

    # Provide additional information
    print("\nSummary:")
    print(f" - Number of inputs: {neuron.num_inputs}")
    print(f" - Weights: {neuron.weights}")
    print(f" - Threshold: {neuron.threshold}")
    print(f" - Input signals: {inputs}")
    print(f" - Weighted sum: {sum(w * i for w, i in zip(neuron.weights, inputs))}")
    print(f" - Neuron output: {'Fired (1)' if output == 1 else 'Did not fire (0)'}")
    print("\nThank you for using the McCulloch-Pitts Neuron Simulator!")
