# Stackelberg Reward Shaping

<p align="center">
  <br />
  <a href="./LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-red.svg" /></a>
  <a href="Python 3.8"><img alt="Python 3.8" src="https://img.shields.io/badge/python-3.8-blue.svg" /></a>
</p>

<!-- ![A brief overview of the Composite Flow Framework.](./imgs/compositeflow.png) -->

🔥 [Reward Shaping for Inferernce-Time Alignment: A Stackelberg Game Perspective](https://arxiv.org/abs/2602.02572)

Stackelberg Reward Shaping studies optimal reward shaping for LLM alignment by formulating the alignment process as a Stackelberg game, or more specifically, a principal–agent model. The reward-model provider acts as the leader, designing a reward function as an incentive mechanism, while the language model acts as the follower and best responds by selecting an aligned policy. 

Our codebase supports reward shaping on two popular inference-time alignment methods: 
1. Alignment as Reward Guided Search (ARGS)
2. Controlled Decoding (CD)

## Run Stackelberg Reward Shaping on ARGS



## Run Stackelberg Reward Shaping on CD


## 📄Citing Stackelberg Reward Shaping
Please consider citing us if you find our work useful!

```
@inproceedings{wang2026reward,
  title={Reward Shaping for Inference-Time Alignment: A Stackelberg Game Perspective},
  author={Wang, Haichuan and Lin, Tao and Kong, Lingkai and Li, Ce and Jiang, Hezi and Tambe, Milind},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  year={2026}
}
```