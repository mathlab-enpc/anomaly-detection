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

## Using a NetLogo model
Beforehand, NetLogo and Java must be installed: 
 - [Install NetLogo](https://ccl.northwestern.edu/netlogo/6.1.1/)
 - [Install Java](https://www.oracle.com/java/technologies/javase-jdk15-downloads.html)

You may need to reinstall PynetLogo and JPype1 after that:
```bash
pip uninstall pynetlogo jpype1 && pip install -r requirements.txt
```

Then, run the following lines to generate the scene:

```bash
python run_netlogo_model.py
python create_video.py --image-folder experiment_images/netlogo_simul --video-filename netlogo_simul_vid
```

You can find a decent guide to NetLogo's language [here](http://ccl.northwestern.edu/netlogo/docs/programming.html#agentsets)


# Troubleshooting
Eventually, PyNetLogo may not be able to detect NetLogo's binaries. In that case, you might see the following error message:

```bash
Traceback (most recent call last):
  File "run_netlogo_model.py", line 10, in <module>
    netlogo = pyNetLogo.NetLogoLink()
  File "/XXXXX/venv/lib/python3.8/site-packages/pyNetLogo/core.py", line 221, in __init__
    netlogo_version = establish_netlogoversion(netlogo_home)
  File "/XXXXX/venv/lib/python3.8/site-packages/pyNetLogo/core.py", line 108, in establish_netlogoversion
    version = match.group()
AttributeError: 'NoneType' object has no attribute 'group'
```

You must find the directory where the binary `NetLogo 6.X.X` is stored and edit the arguments passed to the `NetLogoLink` constructor in the file `run_netlogo_model.py`:

```python
netlogo = pyNetLogo.NetLogoLink(netlogo_home="/path/to/NetLogo/directory/")
```