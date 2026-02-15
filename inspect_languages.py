
import os
import torch
import nemo.collections.asr as nemo_asr

MODEL_PATH = "/app/models/indicconformer_stt_multi_hybrid_rnnt_600m.nemo"

print(f"Loading model from {MODEL_PATH}...")
try:
    if torch.cuda.is_available():
        map_location = "cuda"
    else:
        map_location = "cpu"
        
    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
        MODEL_PATH,
        map_location=map_location,
    )
    
    print("Model loaded.")
    if hasattr(model, 'joint_net'):
        # Only the last layer of joint_net is usually the language specific projection or similar if it's that type
        # The error was: res = self.joint_net[-1][language_ids[0]](inp)
        # So we check keys of model.joint_net[-1]
        print("Available languages in joint_net[-1]:")
        print(list(model.joint_net[-1].keys()))
    else:
        print("Model does not have joint_net attribute similar to expected structure.")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
