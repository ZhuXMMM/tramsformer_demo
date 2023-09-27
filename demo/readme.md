# Transformer Model Implementation

This repository contains a basic implementation of the Transformer model, specifically designed for a unique sentence transformation task.

## File Structure

- `transformer.py`: Contains the implementation of the Transformer model.
- `train.py`: Contains the code for training and testing the Transformer model.

## How to Use

1. Ensure you have PyTorch installed.
2. Run `train.py` to train the model.
3. The training script, `train.py`, leverages the Transformer model defined in `transformer.py`.

## Data

Sample data consists of predefined sentences that map characters (like "a" and "b") to corresponding Chinese characters (like "一" and "二").

## Model

The model is based on the standard Transformer architecture. It has an encoder and a decoder. The encoder reads the input sentence and produces a continuous representation. The decoder then uses this representation to produce the output sentence.

## Acknowledgments

- [this detailed article](https://blog.csdn.net/m0_51474171/article/details/127723423?ops_request_misc=&request_id=&biz_id=102&utm_term=手撕transformer&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-5-127723423.142^v94^chatsearchT3_1&spm=1018.2226.3001.4187)

- ```
  @article{vaswani2017attention,
    title={Attention is all you need},
    author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
    journal={Advances in neural information processing systems},
    volume={30},
    year={2017}
  }
  ```