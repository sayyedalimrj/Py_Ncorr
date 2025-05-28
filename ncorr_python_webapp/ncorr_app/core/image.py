import numpy as np
import cv2 # OpenCV for image loading and GaussianBlur
from scipy.fft import fft, ifft # For B-spline coefficient calculation
# Assuming your C++ bindings are importable like this:
from ncorr_app._ext import _ncorr_cpp_core # For CppNcorrClassImg, CppClassDoubleArray

class NcorrImage:
    """
    Python equivalent of ncorr_class_img.m.
    Handles image data, grayscale conversion, B-spline coefficients, and reduction.
    """
    def __init__(self, source, name="", path=""):
        """
        Initializes NcorrImage.

        Args:
            source (str or np.ndarray): File path to image or a NumPy array.
            name (str, optional): Name of the image. Defaults to "".
            path (str, optional): Path to the image file (if source is a path). Defaults to "".
        """
        self.name = name
        self.path = path
        self.gs_data = None
        self.type = ''
        self._bcoef_data = None
        self.border_bcoef = 20  # Default as in Ncorr
        self.max_gs = 1.0
        self.min_gs = 0.0
        self.height = 0
        self.width = 0

        if isinstance(source, str):
            self.path = source # Assuming source is the full path if name/path not given
            if not self.name:
                self.name = self.path.split('/')[-1].split('\\')[-1] # Basic name extraction
            
            img_bgr = cv2.imread(source)
            if img_bgr is None:
                raise FileNotFoundError(f"Image not found at path: {source}")
            self.type = 'file'
            
            if len(img_bgr.shape) == 3 and img_bgr.shape[2] == 3: # Color image
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            elif len(img_bgr.shape) == 2: # Already grayscale
                img_gray = img_bgr
            else:
                raise ValueError("Unsupported image format. Must be 2D (grayscale) or 3D (BGR).")

            # Normalize to 0.0 - 1.0 float64
            if img_gray.dtype == np.uint8:
                self.gs_data = img_gray.astype(np.float64) / 255.0
            elif img_gray.dtype == np.uint16:
                self.gs_data = img_gray.astype(np.float64) / 65535.0
            elif img_gray.dtype == np.float32 or img_gray.dtype == np.float64:
                # Assuming it's already in a suitable range, or needs normalization
                # For simplicity, let's assume if float, it's already 0-1 or handle as needed
                self.gs_data = img_gray.astype(np.float64)
                if np.max(self.gs_data) > 1.0: # Simple check, might need better normalization
                    self.gs_data /= np.max(self.gs_data)
            else:
                raise ValueError(f"Unsupported image dtype: {img_gray.dtype}")

        elif isinstance(source, np.ndarray):
            if source.ndim == 3 and source.shape[2] == 3: # Color image (assume RGB if from array)
                # Convert RGB to Grayscale: Y = 0.299 R + 0.587 G + 0.114 B
                img_gray = 0.299 * source[:,:,0] + 0.587 * source[:,:,1] + 0.114 * source[:,:,2]
            elif source.ndim == 2: # Already grayscale
                img_gray = source
            else:
                raise ValueError("Unsupported NumPy array format. Must be 2D (grayscale) or 3D (RGB).")

            # Normalize to 0.0 - 1.0 float64
            if np.issubdtype(img_gray.dtype, np.integer):
                max_val = np.iinfo(img_gray.dtype).max
                self.gs_data = img_gray.astype(np.float64) / max_val
            elif np.issubdtype(img_gray.dtype, np.floating):
                self.gs_data = img_gray.astype(np.float64)
                if np.max(self.gs_data) > 1.0: # Simple check
                     self.gs_data /= np.max(self.gs_data)
            else:
                raise ValueError(f"Unsupported NumPy array dtype: {img_gray.dtype}")
            self.type = 'numpy_array'
            if not self.name:
                self.name = "numpy_sourced_image"
        else:
            raise TypeError("Image source must be a file path (str) or a NumPy array.")

        self.height, self.width = self.gs_data.shape
        self.min_gs = np.min(self.gs_data)
        self.max_gs = np.max(self.gs_data) # Ncorr seems to use 0-1 for gs, but sets max_gs often to 1.

    def get_gs(self):
        """Returns the grayscale image data as a NumPy array."""
        return self.gs_data

    def _form_bcoef_py(self, padded_gs_data):
        """
        Python implementation of B-spline coefficient calculation using SciPy FFT.
        Equivalent to ncorr_class_img.form_bcoef(plot_gs) in MATLAB.
        """
        if padded_gs_data.size == 0 or padded_gs_data.shape[0] < 5 or padded_gs_data.shape[1] < 5:
            # Match MATLAB behavior: error if too small, but handle empty
            if padded_gs_data.size == 0: return np.array([])
            raise ValueError("Array for B-spline coefficients must be >= 5x5 or empty.")

        plot_bcoef = np.zeros_like(padded_gs_data, dtype=np.complex128) # Use complex for FFT
        
        kernel_b_1d = np.array([1/120, 13/60, 11/20, 13/60, 1/120], dtype=np.float64)

        # FFT across rows
        kernel_b_x = np.zeros(padded_gs_data.shape[1], dtype=np.float64)
        kernel_b_x[0:3] = kernel_b_1d[2:5]  # [11/20, 13/60, 1/120]
        kernel_b_x[-2:] = kernel_b_1d[0:2] # [1/120, 13/60] (for fftshift-like behavior with MATLAB fft)
        
        fft_kernel_b_x = fft(kernel_b_x)
        
        for i in range(padded_gs_data.shape[0]):
            row_fft = fft(padded_gs_data[i, :])
            # Element-wise division, handle potential division by zero if kernel FFT has zeros
            # (unlikely for this specific B-spline kernel)
            plot_bcoef[i, :] = ifft(row_fft / fft_kernel_b_x)

        # FFT across columns
        kernel_b_y = np.zeros(padded_gs_data.shape[0], dtype=np.float64)
        kernel_b_y[0:3] = kernel_b_1d[2:5]
        kernel_b_y[-2:] = kernel_b_1d[0:2]
        fft_kernel_b_y = fft(kernel_b_y)

        for j in range(padded_gs_data.shape[1]):
            col_fft = fft(plot_bcoef[:, j])
            plot_bcoef[:, j] = ifft(col_fft / fft_kernel_b_y)
            
        return np.real(plot_bcoef) # Result should be real

    def get_bcoef(self):
        """
        Calculates and returns B-spline coefficients for the image.
        Caches the result.
        """
        if self._bcoef_data is None:
            if self.type == 'reduced':
                raise TypeError("B-spline coefficients should not be obtained for reduced images.")
            if self.gs_data is None:
                raise ValueError("Grayscale data (gs_data) must be loaded to calculate B-spline coefficients.")

            padded_gs = np.pad(self.gs_data, self.border_bcoef, mode='edge')
            
            # The original Ncorr pads then fills corners/sides in a specific way.
            # np.pad with mode='edge' is a good approximation. Let's refine padding like Ncorr:
            padded_gs_ncorr = np.zeros((self.height + 2 * self.border_bcoef, self.width + 2 * self.border_bcoef), dtype=np.float64)
            padded_gs_ncorr[self.border_bcoef:-self.border_bcoef, self.border_bcoef:-self.border_bcoef] = self.gs_data
            
            # Fill corners
            padded_gs_ncorr[0:self.border_bcoef, 0:self.border_bcoef] = padded_gs_ncorr[self.border_bcoef, self.border_bcoef] # top-left
            padded_gs_ncorr[0:self.border_bcoef, -self.border_bcoef:] = padded_gs_ncorr[self.border_bcoef, -self.border_bcoef-1] # top-right
            padded_gs_ncorr[-self.border_bcoef:, -self.border_bcoef:] = padded_gs_ncorr[-self.border_bcoef-1, -self.border_bcoef-1] # bottom-right
            padded_gs_ncorr[-self.border_bcoef:, 0:self.border_bcoef] = padded_gs_ncorr[-self.border_bcoef-1, self.border_bcoef] # bottom-left
            
            # Fill sides
            padded_gs_ncorr[0:self.border_bcoef, self.border_bcoef:-self.border_bcoef] = padded_gs_ncorr[self.border_bcoef, self.border_bcoef:-self.border_bcoef] # top
            padded_gs_ncorr[self.border_bcoef:-self.border_bcoef, -self.border_bcoef:] = padded_gs_ncorr[self.border_bcoef:-self.border_bcoef, -self.border_bcoef-1].reshape(-1,1) # right
            padded_gs_ncorr[-self.border_bcoef:, self.border_bcoef:-self.border_bcoef] = padded_gs_ncorr[-self.border_bcoef-1, self.border_bcoef:-self.border_bcoef] # bottom
            padded_gs_ncorr[self.border_bcoef:-self.border_bcoef, 0:self.border_bcoef] = padded_gs_ncorr[self.border_bcoef:-self.border_bcoef, self.border_bcoef].reshape(-1,1) # left

            self._bcoef_data = self._form_bcoef_py(padded_gs_ncorr)
        return self._bcoef_data

    def reduce(self, spacing):
        """
        Reduces the image size for display purposes.

        Args:
            spacing (int): Spacing parameter for reduction.

        Returns:
            NcorrImage: A new NcorrImage instance with the reduced image.
        """
        if self.gs_data is None:
            raise ValueError("Image data not loaded.")

        reduced_gs_data = self.gs_data.copy() # Work on a copy
        if spacing > 0:
            # Low-pass filter (Gaussian blur)
            # Kernel size and sigma based on Ncorr's MATLAB implementation
            kernel_size = (spacing + 1, spacing + 1)
            sigma = (spacing + 1) / 2.0
            reduced_gs_data = cv2.GaussianBlur(reduced_gs_data, kernel_size, sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
            
            # Resample
            reduced_gs_data = reduced_gs_data[::spacing+1, ::spacing+1]

        # Create a new NcorrImage instance for the reduced image
        # We pass the gs_data directly, so the constructor will handle normalization if it's not 0-1
        # However, since we start with gs_data (0-1), blurring should keep it in range.
        reduced_img_obj = NcorrImage(source=reduced_gs_data.copy(), name=self.name + "_reduced", path=self.path)
        reduced_img_obj.type = 'reduced'
        # Reduced images typically don't store B-spline coefficients in Ncorr
        reduced_img_obj._bcoef_data = None 
        return reduced_img_obj

    def to_cpp_img(self):
        """
        Converts this NcorrImage to a CppNcorrClassImg for C++ bindings.
        """
        cpp_img = _ncorr_cpp_core.CppNcorrClassImg()
        cpp_img.type = self.type
        
        if self.gs_data is not None:
            h, w = self.gs_data.shape
            cpp_img.gs.alloc(h, w)
            cpp_img.gs.set_value_numpy(self.gs_data.astype(np.float64)) # Ensure double
        
        bcoef_data = self.get_bcoef() # Calculate if not already done (except for 'reduced')
        if bcoef_data is not None and bcoef_data.size > 0:
            h_bc, w_bc = bcoef_data.shape
            cpp_img.bcoef.alloc(h_bc, w_bc)
            cpp_img.bcoef.set_value_numpy(bcoef_data.astype(np.float64))

        cpp_img.max_gs = float(self.max_gs)
        # min_gs is not a direct member of CppNcorrClassImg in the bindings,
        # but often max_gs and implicit 0 range is used.
        cpp_img.border_bcoef = int(self.border_bcoef)
        # Note: CppNcorrClassImg binding doesn't have height/width, it gets them from gs.
        return cpp_img

    def __repr__(self):
        return f"<NcorrImage name='{self.name}' type='{self.type}' shape=({self.height}, {self.width})>"