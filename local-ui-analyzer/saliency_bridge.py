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
        Load EML-NET model weights from the trained hybrid model.
        """
        print(f"Initializing EML-NET Saliency Engine on {self.device}...")
        
        # Path to the trained hybrid model
        self.hybrid_model_path = os.path.join(self.models_dir, "eml_net_hybrid.pth")
        
        if not os.path.exists(self.hybrid_model_path):
             print(f"Warning: Trained model not found at '{self.hybrid_model_path}'.")
             print("Saliency prediction will fallback to Gaussian blobs.")
             self.model = None
             return
             
        try:
            # Initialize Model
            model = EMLNet()
            
            # Load the unified model weights
            print(f"Loading model from {self.hybrid_model_path}...")
            checkpoint = torch.load(self.hybrid_model_path, map_location=self.device)
            
            # Handle state dict structure
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Handle DataParallel prefix and load
            clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(clean_state_dict, strict=False)
            
            model.to(self.device)
            model.eval()
            self.model = model
            print("EML-NET model loaded successfully.")
            
        except Exception as e:
            print(f"Error loading EML-NET: {e}")
            print("Fallback to Gaussian blobs.")
            self.model = None

    def predict(self, image_numpy):
        """
        Predict saliency map using high-fidelity sliding window inference.
        
        Uses dynamic rescaling for Desktop (1920px) and Mobile (375px) screenshots,
        Lanczos resampling for edge preservation, and 2D Hanning window blending
        with 25% overlap for seam-free tile fusion.
        
        Args:
            image_numpy: Input image (BGR) from OpenCV.
            
        Returns:
            np.ndarray: Saliency map (0-255 uint8)
        """
        if self.model is None:
            return None
            
        try:
            h, w = image_numpy.shape[:2]
            
            # Model expects 4:3 input (640x480)
            model_w, model_h = 640, 480
            
            # --- Dynamic Rescaling ---
            # Desktop (width >= 640): use 640px window
            # Mobile (width < 640): use full image width
            if w >= 640:
                window_w = 640
            else:
                window_w = w
            
            # Maintain 4:3 aspect ratio for EML-NET input
            window_h = int(window_w * (3.0 / 4.0))
            
            # Check if sliding window is needed
            # Use sliding window if image height exceeds a single window
            needs_sliding = h > window_h
            
            if needs_sliding:
                print(f"  High-fidelity sliding window ({w}x{h}). Window: {window_w}x{window_h}")
                
                # --- 25% Overlap Stride ---
                stride_x = int(window_w * 0.75)
                stride_y = int(window_h * 0.75)
                
                # Accumulators in float32 to prevent quantization noise
                full_saliency = np.zeros((h, w), dtype=np.float32)
                weight_accumulator = np.zeros((h, w), dtype=np.float32)
                
                # --- Pre-compute 2D Hanning Window Kernel ---
                hanning_y = np.hanning(window_h).astype(np.float32)
                hanning_x = np.hanning(window_w).astype(np.float32)
                hanning_2d = np.outer(hanning_y, hanning_x)
                
                # --- Sliding Window Loop (both X and Y) ---
                y = 0
                while y < h:
                    # Compute vertical bounds
                    end_y = min(y + window_h, h)
                    start_y = end_y - window_h
                    if start_y < 0:
                        start_y = 0
                        end_y = min(window_h, h)
                    
                    actual_h = end_y - start_y
                    
                    x = 0
                    while x < w:
                        # Compute horizontal bounds
                        end_x = min(x + window_w, w)
                        start_x = end_x - window_w
                        if start_x < 0:
                            start_x = 0
                            end_x = min(window_w, w)
                        
                        actual_w = end_x - start_x
                        
                        # Extract tile
                        tile = image_numpy[start_y:end_y, start_x:end_x]
                        
                        # Predict on tile
                        pred_map = self._predict_single(tile)  # Returns 0-1 float
                        
                        # Resize prediction back to tile size using Lanczos
                        pred_map_resized = cv2.resize(
                            pred_map, 
                            (actual_w, actual_h), 
                            interpolation=cv2.INTER_LANCZOS4
                        )
                        
                        # Get appropriate Hanning window slice
                        if actual_h == window_h and actual_w == window_w:
                            weight_kernel = hanning_2d
                        else:
                            # Recompute for edge tiles with different dimensions
                            edge_hanning_y = np.hanning(actual_h).astype(np.float32)
                            edge_hanning_x = np.hanning(actual_w).astype(np.float32)
                            weight_kernel = np.outer(edge_hanning_y, edge_hanning_x)
                        
                        # Accumulate weighted prediction
                        full_saliency[start_y:end_y, start_x:end_x] += pred_map_resized * weight_kernel
                        weight_accumulator[start_y:end_y, start_x:end_x] += weight_kernel
                        
                        # Move to next horizontal tile
                        if end_x >= w:
                            break
                        x += stride_x
                    
                    # Move to next vertical tile
                    if end_y >= h:
                        break
                    y += stride_y
                
                # --- Final Normalization (single pass to avoid quantization) ---
                final_map = np.divide(
                    full_saliency, 
                    weight_accumulator + 1e-8, 
                    dtype=np.float32
                )
                
            else:
                # Standard inference for images fitting in single window
                final_map = self._predict_single(image_numpy)
                # Resize to original using Lanczos
                final_map = cv2.resize(
                    final_map, (w, h), 
                    interpolation=cv2.INTER_LANCZOS4
                )

            # Normalize final output (0-1 range, then scale to 0-255)
            final_map = (final_map - final_map.min()) / (final_map.max() - final_map.min() + 1e-8)
            return (final_map * 255).astype(np.uint8)

        except Exception as e:
            print(f"Error during saliency prediction: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _predict_single(self, image_numpy):
        """
        Single-pass prediction with Lanczos resampling for high-fidelity downscaling.
        
        Returns 0-1 float map.
        """
        h, w = image_numpy.shape[:2]
        img_rgb = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB)
        
        # --- Lanczos Resampling for Downscaling ---
        # Resize to model input (640x480) using Lanczos to preserve UI edges
        img_resized = cv2.resize(
            img_rgb, 
            (640, 480), 
            interpolation=cv2.INTER_LANCZOS4
        )
        
        # Convert to tensor with normalization
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        img_tensor = transform(img_resized).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            
        saliency_map = output.squeeze().cpu().numpy()  # Raw logits or sigmoid
        
        # Normalize 0-1 locally
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
        
        return saliency_map.astype(np.float32)
