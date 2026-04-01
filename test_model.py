import pytest
import numpy as np
from PIL import Image
import model_engine

def test_preprocess_image_shape():
    test_image = Image.new('RGB', (100, 100), color='red')
    processed = model_engine.preprocess_image(test_image)
    
    assert processed.shape == (1, 32, 32, 3)

def test_preprocess_image_normalization():
    test_image = Image.new('RGB', (32, 32), color=(255, 255, 255))
    processed = model_engine.preprocess_image(test_image)
    
    assert np.max(processed) <= 1.0
    assert np.min(processed) >= 0.0
    assert processed.dtype == np.float32

def test_preprocess_image_error_handling():
    with pytest.raises(ValueError) as excinfo:
        model_engine.preprocess_image("not_an_image_object")
    
    assert "Image processing error" in str(excinfo.value)
