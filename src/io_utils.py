import numpy as np
from astropy.io import fits
import sys


# =============================================================================
# FITS LOADER FUNCTION
# =============================================================================

def load_fits_image(filepath):
    """
    Loads a FITS image, automatically handles BSCALE/BZERO scaling,
    and converts the array into clean float32 format for OpenCV.
    """
    try:
        # 1. Open FITS file and locate valid 2D image data
        with fits.open(filepath, do_not_scale_image_data=False) as hdul:
            data = None
            for hdu in hdul:
                if hdu.data is not None and len(hdu.data.shape) >= 2:
                    data = hdu.data
                    break
            
            if data is None:
                print(f"Warning: No valid image data found in {filepath}")
                return None

        # 2. Clean floating-point anomalies (NaNs and Infinities)
        if data.dtype.kind == 'f':
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # 3. Format conversion to float32 (handling Endianness automatically)
        final_data = data.astype(np.float32)

        return final_data

    except OSError as e:
        print(f"I/O error or corrupted file for {filepath}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error while loading {filepath}: {e}")
        return None