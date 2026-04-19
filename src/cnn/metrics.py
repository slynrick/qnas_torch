import torch
import time

class ModelMetrics:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device

        if next(self.model.parameters()).device != torch.device(self.device):
            self.model.to(self.device)
                    
        if 'cuda' in device and torch.cuda.is_available():
            try:
                import pynvml
                pynvml.nvmlInit()
                self.pynvml = pynvml
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
                self.nvml_initialized = True
            except Exception as e:
                print(f"Error initializing NVML: {e}")
                self.nvml_initialized = False
        else:
            self.nvml_initialized = False

    def __del__(self):
        if hasattr(self, 'nvml_initialized') and self.nvml_initialized:
            self.pynvml.nvmlShutdown()

    def _to_device(self, data):
        if isinstance(data, (list, tuple)):
            return [self._to_device(d) for d in data]
        return data.to(self.device)

    def _check_device(self, data):
        if isinstance(data, (list, tuple)):
            return all(d.device == torch.device(self.device) for d in data)
        return data.device == torch.device(self.device)

    def measure_inference_time(self, input_data, warmup_runs=10, measure_runs=10):
        """
        Measure the inference time of a PyTorch model.
        The function assumes that the model and the input data are on the same device.

        Parameters:
        - input_data: Input data for the model - tensor or list of tensors
        - warmup_runs: Number of warmup runs before measuring
        - measure_runs: Number of runs to measure for averaging

        Returns:
        - average_inference_time: Average inference time in microseconds
        """
        self.model.eval()

        if not self._check_device(input_data):
            input_data = self._to_device(input_data)

        if next(self.model.parameters()).device != torch.device(self.device):
            self.model.to(self.device)

        # Warm up the model
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self.model(input_data)

        # Measure inference time
        inference_times = []
        with torch.no_grad():
            for _ in range(measure_runs):
                if 'cuda' in self.device and torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.time()
                _ = self.model(input_data)
                if 'cuda' in self.device and torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()
                inference_times.append(end_time - start_time)

        average_inference_time = (sum(inference_times) / len(inference_times)) * 1e6  # Convert seconds to microseconds
        return average_inference_time

    def measure_madd(self, input_shape):
        """
        Measure the Multiply-Add operations of a PyTorch model.

        Parameters:
        - input_shape: Shape of the input tensor

        Returns:
        - total_madd: Total Multiply-Add operations
        """
        self.model.eval()

        # Generate random input data
        input_data = torch.randn(input_shape).to(self.device)

        if not self._check_device(input_data):
            input_data = self._to_device(input_data)

        if next(self.model.parameters()).device != torch.device(self.device):
            self.model.to(self.device)

        # Count the Multiply-Add operations
        total_madd = 0
        current_input = input_data

        with torch.no_grad():
            for module in self.model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    # Compute output dimensions
                    out_h = int((current_input.shape[2] + 2 * module.padding[0] - module.dilation[0] * (module.kernel_size[0] - 1) - 1) / module.stride[0] + 1)
                    out_w = int((current_input.shape[3] + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1) / module.stride[1] + 1)
                    madd = module.in_channels * module.out_channels * module.kernel_size[0] * module.kernel_size[1] * out_h * out_w
                    total_madd += madd
                    current_input = torch.zeros((current_input.shape[0], module.out_channels, out_h, out_w)).to(self.device)
                elif isinstance(module, torch.nn.Linear):
                    madd = module.in_features * module.out_features
                    total_madd += madd
                    current_input = torch.zeros((current_input.shape[0], module.out_features)).to(self.device)
                else:
                    # For other layers, the input shape remains the same
                    pass

        return total_madd

    def measure_memory(self, input_shape):
        """
        Measure the memory usage of a PyTorch model.

        Parameters:
        - input_shape: Shape of the input tensor

        Returns:
        - memory: Memory usage in bytes (returns 0 if device is not 'cuda')
        """
        if 'cuda' not in self.device or not torch.cuda.is_available():
            return 0

        self.model.eval()

        # Generate random input data
        input_data = torch.randn(input_shape).to(self.device)

        if not self._check_device(input_data):
            input_data = self._to_device(input_data)

        if next(self.model.parameters()).device != torch.device(self.device):
            self.model.to(self.device)

        # Measure memory usage
        torch.cuda.reset_peak_memory_stats(device=self.device)
        with torch.no_grad():
            _ = self.model(input_data)
        memory = torch.cuda.max_memory_allocated(device=self.device)

        return memory

    def measure_flops(self, input_shape):
        """
        Measure the Floating Point Operations of a PyTorch model.

        Parameters:
        - input_shape: Shape of the input tensor

        Returns:
        - total_flops: Total Floating Point Operations
        """
        self.model.eval()

        # Generate random input data
        input_data = torch.randn(input_shape).to(self.device)

        if not self._check_device(input_data):
            input_data = self._to_device(input_data)

        if next(self.model.parameters()).device != torch.device(self.device):
            self.model.to(self.device)

        # Function to count the Floating Point Operations
        def count_flops(module, input, output):
            flops = 0
            if isinstance(module, torch.nn.Conv2d):
                output_dims = output.shape[2:]
                kernel_dims = module.kernel_size
                in_channels = module.in_channels
                out_channels = module.out_channels
                flops = in_channels * out_channels * kernel_dims[0] * kernel_dims[1] * output_dims[0] * output_dims[1] * 2
            elif isinstance(module, torch.nn.Linear):
                flops = module.in_features * module.out_features * 2
            return flops

        total_flops = 0

        # Register hooks to count FLOPs
        def hook(module, input, output):
            nonlocal total_flops
            total_flops += count_flops(module, input, output)

        hooks = []
        for module in self.model.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                hooks.append(module.register_forward_hook(hook))

        # Perform a forward pass to trigger the hooks
        with torch.no_grad():
            _ = self.model(input_data)

        # Remove hooks
        for h in hooks:
            h.remove()

        return total_flops

    def measure_parameters(self):
        """
        Measure the number of parameters in a PyTorch model.

        Returns:
        - total_params: Total number of parameters
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        return total_params

    def measure_energy_consumption(self, input_data, warmup_runs=10, measure_runs=10):
        """
        Measure the energy consumption of a PyTorch model during inference.

        Parameters:
        - input_data: Input data for the model - tensor or list of tensors
        - warmup_runs: Number of warmup runs before measuring
        - measure_runs: Number of runs to measure for averaging

        Returns:
        - average_power: Average power consumption in watts
        - total_energy: Total energy consumption in joules
        """
        if not self.nvml_initialized:
            print("NVML is not initialized. Cannot measure energy consumption.")
            return None, None

        self.model.eval()

        if not self._check_device(input_data):
            input_data = self._to_device(input_data)

        if next(self.model.parameters()).device != torch.device(self.device):
            self.model.to(self.device)

        # Warm up the model
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self.model(input_data)

        # Measure energy consumption
        power_measurements = []
        total_time = 0
        with torch.no_grad():
            for _ in range(measure_runs):
                if 'cuda' in self.device and torch.cuda.is_available():
                    torch.cuda.synchronize()

                start_time = time.time()
                power_start = self.pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000  # Convert milliwatts to watts
                _ = self.model(input_data)
                if 'cuda' in self.device and torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()
                power_end = self.pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000  # Convert milliwatts to watts

                elapsed_time = end_time - start_time
                total_time += elapsed_time
                average_power = (power_start + power_end) / 2
                energy = average_power * elapsed_time  # Energy in joules
                power_measurements.append(energy)

        total_energy = sum(power_measurements)  # in joules
        average_power = total_energy / total_time if total_time > 0 else 0

        return average_power, total_energy
