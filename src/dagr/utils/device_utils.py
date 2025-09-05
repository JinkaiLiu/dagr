"""
设备管理工具模块
"""
import torch

def get_default_device():
    """获取默认设备"""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def to_device(obj, device=None):
    """将对象移动到指定设备"""
    if device is None:
        device = get_default_device()
    
    if torch.is_tensor(obj):
        return obj.to(device)
    elif hasattr(obj, 'to'):
        return obj.to(device)
    elif hasattr(obj, '__dict__'):
        for attr_name, attr_value in obj.__dict__.items():
            if torch.is_tensor(attr_value):
                setattr(obj, attr_name, attr_value.to(device))
        return obj
    else:
        return obj

def ensure_same_device(model, *args):
    """确保模型和所有参数在同一设备上"""
    device = next(model.parameters()).device
    
    results = []
    for arg in args:
        results.append(to_device(arg, device))
    
    return results if len(results) > 1 else results[0]

class DeviceManager:
    """设备管理器"""
    def __init__(self, device=None):
        self.device = device or get_default_device()
    
    def __call__(self, obj):
        return to_device(obj, self.device)
    
    def setup_model(self, model):
        """设置模型到指定设备"""
        model = model.to(self.device)
        # 确保所有参数都需要梯度（如果原来需要的话）
        return model

# 全局设备管理器实例
device_manager = DeviceManager()
