import torch

class HookModule(torch.nn.Module):
    def __init__(self, module, height, width, input_channels=3, feature_layers=(), output_layers=(), feature_channels=None, output_channels=None):
        torch.nn.Module.__init__(self)
        
        # 首先确保模块在正确的设备上
        if torch.cuda.is_available():
            self.module = module.cuda()
        else:
            self.module = module
        
        # 替换有问题的 fc 层
        if hasattr(self.module, 'fc') and isinstance(self.module.fc, torch.nn.Linear):
            self.replace_fc_layer()
        
        if input_channels != 3:
            old_conv1 = self.module.conv1
            self.module.conv1 = torch.nn.Conv2d(
                in_channels=input_channels, 
                out_channels=old_conv1.out_channels,
                kernel_size=old_conv1.kernel_size,
                stride=old_conv1.stride,
                padding=old_conv1.padding,
                bias=old_conv1.bias is not None
            )
            
            # 确保新的 conv1 在正确的设备上
            if torch.cuda.is_available():
                self.module.conv1 = self.module.conv1.cuda()
        
        self.feature_layers = feature_layers
        self.output_layers = output_layers
        self.hooks = []
        self.features = []
        self.outputs = []
        self.register_hooks()
        
        self.feature_channels = []
        self.output_channels = []
        self.compute_channels_with_dummy(shape=(1, input_channels, height, width))
        
        self.feature_dconv = torch.nn.ModuleList()
        if feature_channels is not None:
            assert len(feature_channels) == len(self.feature_channels)
            self.feature_dconv = torch.nn.ModuleList(
                [
                    torch.nn.Conv2d(in_channels=cin, out_channels=cout, kernel_size=1, stride=1, padding=0)
                    for cin, cout in zip(self.feature_channels, feature_channels)
                ]
            )
            self.feature_channels = feature_channels
            if torch.cuda.is_available():
                self.feature_dconv = self.feature_dconv.cuda()
        
        self.output_dconv = torch.nn.ModuleList()
        if output_channels is not None:
            assert len(output_channels) == len(self.output_channels)
            self.output_dconv = torch.nn.ModuleList(
                [
                    torch.nn.Conv2d(in_channels=cin, out_channels=cout, kernel_size=1, stride=1, padding=0)
                    for cin, cout in zip(self.output_channels, output_channels)
                ]
            )
            self.output_channels = output_channels
            if torch.cuda.is_available():
                self.output_dconv = self.output_dconv.cuda()

    def replace_fc_layer(self):
        """创建一个完全避免 CUBLAS 的 fc 层"""
        print("Creating CPU-based fc layer to avoid CUBLAS issue")
        
        original_fc = self.module.fc
        
        class CPULinear(torch.nn.Module):
            def __init__(self, original_linear):
                super().__init__()
                # 使用 register_buffer 确保参数保持在 CPU 上
                self.register_buffer('weight', original_linear.weight.data.cpu())
                if original_linear.bias is not None:
                    self.register_buffer('bias', original_linear.bias.data.cpu())
                else:
                    self.register_buffer('bias', None)
                
            def forward(self, x):
                # 确保输入是 2D
                if len(x.shape) > 2:
                    x = x.view(x.size(0), -1)
                
                # 在 CPU 上计算
                x_cpu = x.cpu()
                
                # 确保权重在 CPU 上
                weight_cpu = self.weight.cpu() if self.weight.is_cuda else self.weight
                bias_cpu = self.bias.cpu() if self.bias is not None and self.bias.is_cuda else self.bias
                
                output_cpu = torch.nn.functional.linear(x_cpu, weight_cpu, bias_cpu)
                
                # 移回原设备
                return output_cpu.cuda() if x.is_cuda else output_cpu
            
            def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
                # 自定义加载逻辑以兼容原始参数名
                weight_key = prefix + 'weight'
                bias_key = prefix + 'bias'
                
                if weight_key in state_dict:
                    self.weight = state_dict[weight_key].cpu()
                if bias_key in state_dict:
                    self.bias = state_dict[bias_key].cpu()
                
                # 移除这些键以避免意外的键错误
                if weight_key in state_dict:
                    del state_dict[weight_key]
                if bias_key in state_dict:
                    del state_dict[bias_key]
        
        # 替换为 CPU 版本
        cpu_fc = CPULinear(original_fc)
        self.module.fc = cpu_fc
        print("Successfully created CPU-based fc layer with compatible parameters")

    def extract_layer(self, module, layer):
        if len(layer) == 0:
            return module
        else:
            return self.extract_layer(module._modules[layer[0]], layer[1:])

    def compute_channels_with_dummy(self, shape):
        print(f"Computing channels with dummy input shape: {shape}")
        
        # 创建 dummy input 并确保在正确设备上
        dummy_input = torch.zeros(shape)
        
        # 检查模块设备
        try:
            module_device = next(self.module.parameters()).device
            dummy_input = dummy_input.to(module_device)
            print(f"Dummy input device: {dummy_input.device}, Module device: {module_device}")
        except StopIteration:
            print("No parameters found in module")
        
        # 确保模块完全在正确设备上
        if torch.cuda.is_available():
            self.module = self.module.cuda()
            dummy_input = dummy_input.cuda()
        
        with torch.no_grad():
            try:
                self.module.forward(dummy_input)
                print("Dummy forward pass successful")
            except Exception as e:
                print(f"Error during dummy forward: {e}")
                raise e
        
        self.feature_channels = [f.shape[1] for f in self.features]
        self.output_channels = [o.shape[1] for o in self.outputs]
        print(f"Feature channels: {self.feature_channels}")
        print(f"Output channels: {self.output_channels}")
        
        self.features = []
        self.outputs = []

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

    def register_hooks(self):
        self.features = []
        self.outputs = []
        features_hook = lambda m, i, o: self.features.append(o)
        outputs_hook = lambda m, i, o: self.outputs.append(o)
        for l in self.feature_layers:
            hook_id = self.extract_layer(self.module, l.split(".")).register_forward_hook(features_hook)
            self.hooks.append(hook_id)
        for l in self.output_layers:
            hook_id = self.extract_layer(self.module, l.split(".")).register_forward_hook(outputs_hook)
            self.hooks.append(hook_id)

    def forward(self, x):
        # 确保输入和模块在同一设备上
        try:
            module_device = next(self.module.parameters()).device
            x = x.to(module_device)
        except StopIteration:
            pass
            
        self.features = []
        self.outputs = []
        
        # 直接使用模块的 forward，现在使用包装的 fc 层
        main_output = self.module(x)
        
        features = self.features
        if len(self.feature_dconv) > 0:
            features = [dconv(f) for f, dconv in zip(self.features, self.feature_dconv)]
        
        outputs = self.outputs
        if len(self.output_dconv) > 0:
            outputs = [dconv(o) for o, dconv in zip(self.outputs, self.output_dconv)]
        
        return features, outputs
