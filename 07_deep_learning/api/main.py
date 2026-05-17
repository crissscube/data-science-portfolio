from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import model as digit_model

ml_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model once at startup — not on every request
    ml_model['net'] = digit_model.load_model('digit_cnn_checkpoint.pth')
    print('Model loaded successfully.')
    yield
    ml_model.clear()


app = FastAPI(
    title='Digit Recognition API',
    description='CNN trained on MNIST — predicts handwritten digits (0–9) from images.',
    version='1.0.0',
    lifespan=lifespan
)


@app.get('/')
def root():
    return {
        'name': 'Digit Recognition API',
        'model': 'DigitCNN (3 conv blocks + classifier)',
        'dataset': 'MNIST',
        'test_accuracy': '99.5%+',
        'usage': 'POST /predict with an image file'
    }


@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': 'net' in ml_model}


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ('image/png', 'image/jpeg', 'image/jpg'):
        raise HTTPException(status_code=400, detail='Only PNG and JPEG images are supported.')

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail='Empty file.')

    result = digit_model.predict(ml_model['net'], image_bytes)

    return JSONResponse(content={
        'filename': file.filename,
        'predicted_digit': result['digit'],
        'confidence': f"{result['confidence']:.2%}",
        'all_probabilities': result['probabilities']
    })
