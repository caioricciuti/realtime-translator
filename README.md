# Real-Time Translator

The Real-Time Translator is a Python-based application that provides real-time translation of spoken language. It utilizes the Whisper model for speech-to-text transcription and the M2M100 model for language translation. The application also monitors system resource usage to ensure efficient performance.

## Features

- Real-time audio transcription and translation
- Supports multiple languages
- Speaker identification
- Resource usage monitoring (CPU, Memory, Disk, Network)
- Configurable resource usage limits

## Requirements

- Python 3.7 or higher
- Required Python packages:
  - `sounddevice`
  - `numpy`
  - `transformers`
  - `whisper`
  - `queue`
  - `threading`
  - `customtkinter`
  - `torch`
  - `datetime`
  - `time`
  - `os`
  - `scipy`
  - `sklearn`
  - `psutil`
  - `matplotlib`
  - `tkinter`

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/realtime-translator.git
   cd realtime-translator
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Download the necessary models:

   ```bash
   # This will automatically download the models when you run the script
   ```

## Usage

1. Run the application:

   ```bash
   python translator_app.py
   ```

2. Use the graphical interface to select source and target languages, start and stop translation, and monitor resource usage.

## Configuration

- CPU and Memory usage limits can be configured in the `Config` tab.
- Resource usage statistics are available in the `Resources` tab.

## License

This project is licensed under the MIT License.

## Acknowledgements

- [Whisper](https://github.com/openai/whisper)
- [M2M100](https://huggingface.co/facebook/m2m100_418M)
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)

# Credits

This project makes use of the following open-source libraries and tools:

- **sounddevice**

  - **License**: MIT License
  - **Repository**: [python-sounddevice](https://github.com/spatialaudio/python-sounddevice)

- **numpy**

  - **License**: BSD License
  - **Repository**: [numpy](https://github.com/numpy/numpy)

- **transformers**

  - **License**: Apache 2.0 License
  - **Repository**: [transformers](https://github.com/huggingface/transformers)

- **whisper**

  - **License**: MIT License
  - **Repository**: [whisper](https://github.com/openai/whisper)

- **queue** (Python Standard Library)

  - **License**: Python Software Foundation License
  - **Documentation**: [queue](https://docs.python.org/3/library/queue.html)

- **threading** (Python Standard Library)

  - **License**: Python Software Foundation License
  - **Documentation**: [threading](https://docs.python.org/3/library/threading.html)

- **customtkinter**

  - **License**: MIT License
  - **Repository**: [customtkinter](https://github.com/TomSchimansky/CustomTkinter)

- **torch**

  - **License**: BSD License
  - **Repository**: [pytorch](https://github.com/pytorch/pytorch)

- **datetime** (Python Standard Library)

  - **License**: Python Software Foundation License
  - **Documentation**: [datetime](https://docs.python.org/3/library/datetime.html)

- **time** (Python Standard Library)

  - **License**: Python Software Foundation License
  - **Documentation**: [time](https://docs.python.org/3/library/time.html)

- **os** (Python Standard Library)

  - **License**: Python Software Foundation License
  - **Documentation**: [os](https://docs.python.org/3/library/os.html)

- **scipy**

  - **License**: BSD License
  - **Repository**: [scipy](https://github.com/scipy/scipy)

- **sklearn (scikit-learn)**

  - **License**: BSD License
  - **Repository**: [scikit-learn](https://github.com/scikit-learn/scikit-learn)

- **psutil**

  - **License**: BSD License
  - **Repository**: [psutil](https://github.com/giampaolo/psutil)

- **matplotlib**

  - **License**: Matplotlib License (BSD compatible)
  - **Repository**: [matplotlib](https://github.com/matplotlib/matplotlib)

- **tkinter** (Python Standard Library)
  - **License**: Python Software Foundation License
  - **Documentation**: [tkinter](https://docs.python.org/3/library/tkinter.html)

We gratefully acknowledge the contributions of the developers and maintainers of these libraries.

