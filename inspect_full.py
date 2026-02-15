
import torch
import nemo.collections.asr as nemo_asr
import os

MODEL_PATH = "/app/models/indicconformer_stt_multi_hybrid_rnnt_600m.nemo"

try:
    if torch.cuda.is_available():
        map_location = "cuda"
    else:
        map_location = "cpu"
        
    print(f"Loading {MODEL_PATH} on {map_location}...")
    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
        MODEL_PATH,
        map_location=map_location,
    )
    
    # Try to access joint_net directly via finding modules
    print("\nAttempting to inspect joint_net...")
    if hasattr(model, 'joint_net'):
        jn = model.joint_net
        print(f"joint_net type: {type(jn)}")
        if isinstance(jn, torch.nn.Sequential) or isinstance(jn, torch.nn.ModuleList):
            last_layer = jn[-1]
            print(f"Last layer type: {type(last_layer)}")
            if hasattr(last_layer, 'keys'):
                 print("Keys:", list(last_layer.keys()))
            else:
                 print("Last layer does not have .keys()")
                 print(last_layer)
        else:
             print("joint_net is not Sequential/List")
             print(jn)
    else:
        print("Model does not have 'joint_net' attribute directly.")
        print("Model keys:", list(model.__dict__.keys()))

except Exception as e:
    print(f"Error: {e}")
