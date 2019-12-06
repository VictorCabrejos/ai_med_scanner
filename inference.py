
import torch
from commons import get_model,get_tensor

class_names=['PNEUMONIA', 'NORMAL']
model=get_model()
def get_call_name(image_bytes):
    tensor=get_tensor(image_bytes)
    outputs=model(tensor)
    _,prediction=outputs.max(1)
    category=prediction.item()
    call_name=class_names[category]
    return call_name
    