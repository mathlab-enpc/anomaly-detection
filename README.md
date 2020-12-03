# Crowd anomaly detection

## Installation
Create a virtual environment:
```bash
python3 -m venv ./venv
source venv/bin/activate
```

Run:
```bash
pip install -r requirements.txt
```

## Usage
To activate your virtual environment, run:
```bash
source venv/bin/activate
```
you can quit this virtual environment by running `deactivate` in your terminal.

Then, generate crowd scenarios:
```bash
python main.py
```

The scenarios are stored in the `experiment_images` directory. Then, you can compute the density for each frame of a specific scenario (here `translated_sequence_gathering`):

```bash
python density_computation.py --image-folder experiment_images/translated_sequence_gathering --output-folder density_experiment/translated_sequence_gathering
```

To compute the speed vector field, run:
```bash
python speed_computation.py --image-folder experiment_images/translated_sequence_gathering --output-folder speed_experiment/translated_sequence_gathering
```

To compute the derivative of the density w.r.t time, run:
```bash
python heatmap_time_derivative.py --image-folder experiment_images/translated_sequence_gathering --output-folder density_first_derivative/translated_sequence_gathering
```

Finally, to generate a video with a sequence of frame, run:
```bash
python create_video.py --image-folder experiment_images/translated_sequence_gathering --video-filename my_custom_video_filename
```

It will generate the file `videos/my_custom_video_filename.mp4` with the sequence of frames stored in `experiment_images/translated_sequence_gathering`.