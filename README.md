# SEA
Repository for the paper titled " SEA: Low-Resource Safety Alignment for Multimodal Large Language Models via Synthetic Embeddings"

## SEA
The code and data are coming soon.


## VA-SafetyBench
VA-SafetyBench is a safety benchmark designed to evaluate image-based and video-based MLLMs. It modifies the textual prompts from MM-SafetyBench and uses text-based generative models to produce video and audio data.

### Start the evaluation
1. download the video and audio files from [VA-SafetyBench](https://huggingface.co/datasets/luweikai/VA-SafetyBench).
2. Unzip the video.zip and audio.zip.
3. Use your path to modify lines 181-184 in evaluate_audio.py, as well as lines 185-188 in evaluate_audio.py. Also, set your api_key in the get_GPT_res function in both files.
4. Execute the following commands:
```bash
cd ./VA-SafetyBench
python evaluate_video.py
python evaluate_audio.py
```

### License
VA-SafetyBench is released under CC BY NC 4.0. They are also restricted to uses that follow the license agreement [MM-SafetyBench](https://github.com/isxinliu/mm-safetybench), [Pyramid Flow](https://huggingface.co/rain1011/pyramid-flow-miniflux) and Edge-tts.


##  Acknowledgments
We sincerely thank MM-SafetyBench, Pyramid Flow and Edge-tts as VA-SafetyBench is built upon the foundation of their work.
