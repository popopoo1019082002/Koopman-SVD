import torch
import torch.nn as nn

class KoopmanOperator(nn.Module):
    def __init__(self, dynamics_matrix):
        """
        Initialize the Koopman operator with a fixed linear dynamics matrix.
        
        Args:
            dynamics_matrix (torch.Tensor): A 2D tensor (matrix) representing the dynamics.
        """
        super(KoopmanOperator, self).__init__()
        
        # Store the dynamics matrix as a non-trainable parameter
        # In real applications, this might be learned, but for this setup, it is fixed
        self.dynamics_matrix = nn.Parameter(dynamics_matrix, requires_grad=False)

    def forward(self, x):
        """
        Apply the Koopman operator to input state x using the dynamics matrix.
        
        Args:
            x (torch.Tensor): Input state vector(s).
        
        Returns:
            torch.Tensor: Transformed state(s) after applying the Koopman dynamics.
        """
        # Perform matrix multiplication to transform input state according to dynamics
        return torch.matmul(x, self.dynamics_matrix)


# Example setup for testing:
if __name__ == "__main__":
    # Define a simple 3x3 linear dynamics matrix
    dynamics_matrix = torch.tensor([
        [0.9, 0.1, 0.0],
        [-0.1, 0.9, 0.1],
        [0.0, -0.1, 0.9]
    ], dtype=torch.float32)

    # Instantiate the KoopmanOperator with this dynamics matrix
    koopman_operator = KoopmanOperator(dynamics_matrix)

    # Sample input for testing (identity matrix for simplicity)
    x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)

    # Apply the Koopman operator and print the transformed state
    transformed_state = koopman_operator(x)
    print("Transformed State:\n", transformed_state)
