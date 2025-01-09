from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import keras
import numpy as np
import uvicorn

app = FastAPI()

@app.post("/classify")
async def classify_image(image_file: UploadFile = File(...)):
    try:
        if not image_file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

        # Load the ResNet50 model
        model = keras.applications.ResNet50(weights="imagenet")
        # Load the image
        image = keras.preprocessing.image.load_img(image_file.filename, target_size = (224, 224))
        # Convert image to array and preprocess
        img_array = keras.preprocessing.image.img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = keras.applications.resnet50.preprocess_input(img_array)

        predictions = model.predict(img_array)
        # Format result
        result = [
            {"description": pred[0], "class": pred[1], "probability": float(pred[2])}
            for sublist in keras.applications.resnet50.decode_predictions(predictions)
            for pred in sublist
        ]
        return JSONResponse(content={"predictions": result})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


if __name__ == "__main__":
    uvicorn.run(app,
                host="127.0.0.1",
                port=8000)