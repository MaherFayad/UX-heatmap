import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

# Import the model architecture
try:
    from eml_net_model import EMLNet
except ImportError:
    # Handle case where file might not be found if running from different context
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from eml_net_model import EMLNet

class SaliencyEngine:
    def __init__(self, models_dir="models"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.models_dir = models_dir
        
        # Define paths
        self.encoder_path = os.path.join(models_dir, "resnet50_places365.pth.tar")
        self.decoder_path = os.path.join(models_dir, "eml_net_decoder.pth")
        
        self.load_model()
        
    def load_model(self):
        """
        Load EML-NET model weights.
        If weights are missing, warnings are printed and self.model remains None.
        """
        print(f"Initializing EML-NET Saliency Engine on {self.device}...")
        
        # Check if model files exist
        if not os.path.exists(self.models_dir):
            print(f"Warning: Models directory '{self.models_dir}' not found.")
            print("Please create it and download 'resnet50_places365.pth.tar' and 'eml_net_decoder.pth'.")
            return

        if not os.path.exists(self.decoder_path):
             print(f"Warning: EML-NET decoder weights not found at '{self.decoder_path}'.")
             print("Saliency prediction will fallback to Gaussian blobs.")
             return
             
        try:
            # Initialize Model
            model = EMLNet()
            
            # 1. Load Encoder Weights (ResNet50 Places365)
            # This is optional but recommended for better accuracy. 
            # If missing, it uses standard ImageNet weights initiated in EMLNet class.
            if os.path.exists(self.encoder_path):
                print("Loading Encoder (Places365)...")
                # Note: The loading logic depends on how the .tar is structured.
                # Usually it's 'state_dict'. We'll assume standard format.
                checkpoint = torch.load(self.encoder_path, map_location='cpu')
                
                # Check if 'state_dict' key exists
                state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
                
                # Filter out keys that might not match exactly or handle prefix 'module.'
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k.replace("module.", "")
                    # Match only encoder layers
                    if name in model.state_dict():
                        new_state_dict[name] = v
                
                model.load_state_dict(new_state_dict, strict=False)
            else:
                print("Warning: Encoder weights not found. Using default backbone weights.")

            # 2. Load Decoder Weights (EML-NET specific)
            print("Loading Decoder...")
            decoder_checkpoint = torch.load(self.decoder_path, map_location='cpu')
            # Assuming the decoder file contains the state dict for the decoder parts
            # or the whole model. EML-NET repo usually saves the whole model or decoder.
            # We'll try to load strictly first, then non-strictly.
            
            # Handle potential DataParallel wrapping 'module.' prefix
            decoder_state = decoder_checkpoint['state_dict'] if 'state_dict' in decoder_checkpoint else decoder_checkpoint
            clean_decoder_state = {k.replace("module.", ""): v for k, v in decoder_state.items()}
            
            model.load_state_dict(clean_decoder_state, strict=False)
            
            model.to(self.device)
            model.eval()
            self.model = model
            print("EML-NET loaded successfully.")
            
        except Exception as e:
            print(f"Error loading EML-NET: {e}")
            print("Fallback to Gaussian blobs.")
            self.model = None

    def predict(self, image_numpy):
        """
        Predict saliency map for a given image.
        
        Args:
            image_numpy: Input image (BGR) from OpenCV.
            
        Returns:
            np.ndarray: Saliency map (0-255, dimensions match input) or None if model not loaded.
        """
        if self.model is None:
            return None
            
        try:
            # Preprocessing
            # 1. Convert BGR to RGB
            img_rgb = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            
            # 2. Transform pipeline
            # Standard ImageNet normalization + Resize
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((480, 640)), # Fixed input size for model
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            img_tensor = transform(img_rgb).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                output = self.model(img_tensor)
                
            # Postprocessing
            # Convert to numpy
            saliency_map = output.squeeze().cpu().numpy()
            
            # Resize back to original image size
            saliency_map = cv2.resize(saliency_map, (w, h), interpolation=cv2.INTER_CUBIC)
            
            # Normalize to 0-1
            saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
            
            # Convert to 0-255 uint8 if needed just for visualization, 
            # but usually we want float for calculations. 
            # The prompt asks for "A 0-255 grayscale heatmap".
            saliency_map_uint8 = (saliency_map * 255).astype(np.uint8)
            
            return saliency_map_uint8
            
        except Exception as e:
            print(f"Error during saliency prediction: {e}")
            return None
