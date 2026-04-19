# fitness_utils.py

def mofitness(metric_value, params, inference_time, T_p, T_t, metric_type='accuracy') -> float:
    """
    Weighted fitness function for multi-objective optimization.

    :param metric_value: Classification accuracy (0 to 100) or loss of the architecture
    :param params: Number of parameters of the architecture
    :param inference_time: Inference time of the architecture
    :param T_p: Maximum allowable number of parameters
    :param T_t: Maximum allowable inference time
    :param metric_type: Type of the primary metric ('accuracy' or 'loss')
    :return: Fitness value
    """
    # Handle the primary metric based on its type
    if metric_type == 'accuracy':
        # Ensure accuracy is between 0 and 1
        if metric_value > 1:
            metric_value /= 100.0
        primary_fitness = metric_value  # Higher accuracy leads to higher fitness
    elif metric_type == 'loss':
        # Transform loss to a fitness value
        loss = metric_value
        primary_fitness = 1 / (1 + loss)  # Lower loss leads to higher fitness
    else:
        raise ValueError(f"Invalid metric_type: {metric_type}. Choose 'accuracy' or 'loss'.")

    # Assign weights based on thresholds
    w_p = -0.01 if params <= T_p else -1
    w_t = -0.01 if inference_time <= T_t else -1

    params_ratio = params / T_p if T_p != 0 else float('inf')
    inference_time_ratio = inference_time / T_t if T_t != 0 else float('inf')

    # Calculate factors
    params_factor = params_ratio ** w_p if params_ratio != 0 else 0
    inference_time_factor = inference_time_ratio ** w_t if inference_time_ratio != 0 else 0

    fitness_value = primary_fitness * params_factor * inference_time_factor # Weighted fitness value

    return fitness_value * 100.0  # Scale fitness value to 0-100

