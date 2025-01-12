## Pothole Detection
This project uses a YOLOv8 model to detect potholes in images and videos.

## Project Structure
results = model.predict(source="path/to/image.jpg", save=True, conf=0.25)
```bash
Pothole-detection/
├── artifacts/             # Sample videos and images to test
├── results/               # Directory to save the output images and videos
├── weights/               # Contains yolo model weights
├── main.py                # Main script for running the detection
├── helper.py              # helper functions
├── experiments.ipynb      # Jupyter notebook contains step by step training implementation and result metrics
└── requirements.txt       # required packages
```

## Usage
1. Clone repo
```
git clone https://github.com/amaanrzv39/Pothole-detection.git
cd Pothole-detection
```
2. Setup virtual env
```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
3. Install required packages
```
pip install -r requirements.txt
```
4. Run
```
python main.py --source <path_to_image_or_video> --conf <confidence_threshold>
```

## Results on test images
![Unknown-2](https://github.com/user-attachments/assets/2bd410d1-8614-4be8-9692-1a9da182a6e9)

![Unknown-3](https://github.com/user-attachments/assets/48012440-db20-4852-b34b-a7a83caa92b3)

![Unknown-5](https://github.com/user-attachments/assets/7a41c9b6-8ec4-4fc2-83e4-54967247268f)

## Contributing
Contributions are welcome! If you have ideas or encounter issues, feel free to open a pull request or an issue.

## License
This project is licensed under the MIT License.
