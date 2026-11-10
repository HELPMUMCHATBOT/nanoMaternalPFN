"""Export full-context inference (not a KV-cached graph). Requires onnx."""
from pathlib import Path
import torch

class ExportWrapper(torch.nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model=model
    def forward(self,x,y_train):
        return self.model(x,y_train)

def export(model, output, n_context=16, n_query=1, n_features=6, opset=18):
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError('install optional edge dependency: pip install -e .[edge]') from exc
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    model.eval().cpu()
    x=torch.randn(1,n_context+n_query,n_features)
    y=torch.zeros(1,n_context,dtype=torch.long)
    torch.onnx.export(ExportWrapper(model),(x,y),str(output),opset_version=opset,input_names=['x','y_train'],output_names=['logits'],dynamo=False)
    return output
