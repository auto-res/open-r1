
## Setup
### Build the docker environment
There are two ways to build the docker environment.
#### 1. Build the docker environment with the following command
```shell
docker build --no-cache -t aropenr1 .
docker run -v $(pwd):/app -v /work/bmb/hf_models:/work/bmb/hf_models --shm-size=1024g --gpus all --name AROPENR1 -it aropenr1
```
#### 2. Build the container with `compose.yml`, and then enter the container with vscode
```shell
cp .env.example .env
```
After editting `.env`, run:

```shell
docker compose up -d --build
```
Finally you can enter the running container via vscode.
For more details, see [link](https://qiita.com/nonamenme/items/43243586c81cbbb9b08b#2-%E3%82%B3%E3%83%B3%E3%83%86%E3%83%8A%E3%81%B8%E3%82%A2%E3%82%BF%E3%83%83%E3%83%81 ).

### After entering the container
```shell
uv venv openr1 --python 3.11 && source openr1/bin/activate && uv pip install --upgrade pip
uv pip install vllm==0.7.2
uv pip install setuptools && uv pip install flash-attn --no-build-isolation
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ".[dev]"
huggingface-cli login
wandb login
```

## Training models

We support training models with either DDP or DeepSpeed (ZeRO-2 and ZeRO-3). For example, to run SFT on a dataset distilled from DeepSeek-R1 with reasoning traces such as [open-r1/OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k), run:

```shell
# Train via command line
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name open-r1/OpenR1-Math-220k \
    --learning_rate 1.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 16384 \
    --per_device_train_batch_size 16 \
    --gradient_checkpointing \
    --bf16 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill

# Train via YAML config
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --config recipes/Qwen2.5-1.5B-Instruct/sft/config_demo.yaml
```

Currently, the following tasks are supported:

* Supervised Fine-Tuning `sft`
* Group Relative Policy Optimization `grpo`

> [!TIP]
> If you scale up/down the number of GPUs, we recommend also scaling up the per-device batch size or number of gradient accumulation steps to keep the global batch size constant.

By default, these scripts will push each model to your Hugging Face Hub username, i.e. `{username}/{model_name}-{task}`. You can override the parameters in each YAML config by appending them to the command as follows: 

```shell
# Change batch size, number of epochs etc
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --config recipes/Qwen2.5-1.5B-Instruct/sft/config_demo.yaml
    --per_device_train_batch_size=1 --num_train_epochs=5
```

If you also wish to override the Weights and Biases default settings, you can do so as follows:

```shell
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --config recipes/Qwen2.5-1.5B-Instruct/sft/config_demo.yaml
    --wandb_entity huggingface --wandb_project open-r1 --run_name Qwen2.5-1.5B-GRPO
```

> [!NOTE]
> The training commands below are configured for a node of 8 x H100s (80GB). For different hardware and topologies, you may need to tune the batch size and number of gradient accumulation steps.

### SFT

To run SFT on a dataset distilled from DeepSeek-R1 with reasoning traces such as [open-r1/OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k), run:

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/sft.py \
    --config recipes/Qwen2.5-1.5B-Instruct/sft/config_demo.yaml
```

### GRPO

To train via the GRPO trainer, we use one GPU to run vLLM for faster generation and the remaining GPUs for training. For example, one a node with 8 GPUs, set `--num_processes` to override the default value in the `accelerate` configs:

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=7 src/open_r1/grpo.py \
    --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml
```

> [!WARNING]
> The chat template used in the distilled DeepSeek models omits the contents of the reasoning block within the `<think>` and `</think>` tags. It also prefills the assistant response with `<think>` which interferes with the format reward function. To handle that, it is important to override the chat template as done in e.g.  [recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml](./recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml).


We provide a minimal reproducible experiment using GRPO for mathematical reasoning, referencing the approach from [SimpleRL-Reason](https://hkust-nlp.notion.site/simplerl-reason) which uses a 7B model trained on 8K examples. Running this on 8 H100 80G GPU takes about 3 hours:

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=7 src/open_r1/grpo.py \
    --config recipes/Qwen2.5-Math-7B/grpo/config_simple_rl.yaml
```

Our final [model](https://huggingface.co/Dongwei/Qwen-2.5-7B_Base_Math_smalllr), while using different learning rates, loss functions and reward structures, achieves 69.4% accuracy on MATH-500, demonstrating a 17%+ improvement over the base model.

#### 👨‍💻 Training with a code interpreter

We provide a `code` reward function for executing code generated by the policy during training. Currently, this reward function targets code contests like [Codeforces](https://codeforces.com), where solutions are executed against a set of test cases and the overall success rate is returned as the final reward. To ensure safe execution, we use [E2B](https://e2b.dev) sandboxes, which are fast and cheap to run. To use this reward function, first install the necessary dependencies:

```shell
uv pip install -e '.[code]'
```

Then create a `.env` file and place an API token from E2B within it:

```
E2B_API_KEY="e2b_xxx"
```

Then make sure your dataset contains a `verification_info` column with the following schema (adopted from PrimeIntellect's excellent [datasets](https://huggingface.co/collections/PrimeIntellect/synthetic-1-67a2c399cfdd6c9f7fae0c37) of verifiable problems):

```python
{
    "language": "python",
    "test_cases": [
        {
            "input": "4\n4\n0001\n1000\n0011\n0111\n3\n010\n101\n0\n2\n00000\n00001\n4\n01\n001\n0001\n00001\n",
            "output": "1\n3 \n-1\n0\n\n2\n1 2 \n",
            "type": "stdin_stdout",
        }
    ],
}
```

For example, to train a smol model on Python problems, run:

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=7 src/open_r1/grpo.py \
    --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo_code.yaml
```

#### Data decontamination

Following [s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393) the data can be decontaminated using the script at: [scripts/decontaminate.py](./scripts/decontaminate.py), which decontaminates a dataset using 8-grams and deduplicate the data. Sample run:

```shell
python scripts/decontaminate.py \
    --dataset "open-r1/verifiable-coding-problems-python" \
    --problem_column problem \
    --cleanup
```

It will decontaminate against the benchmark datasets, and remove the contaminated samples afterwards. If no argument `--new_dataset_name` is provided, the same dataset will be reused, adding a `_decontaminated`. It runs against the prompt, which for this dataset is the column `problem`, but a different one can be provided.

Arguments for the script:

```shell
usage: decontaminate.py [-h] --dataset DATASET [--split SPLIT] [--ngram_size NGRAM_SIZE] [--problem_column PROBLEM_COLUMN] [--cleanup] [--new_dataset_name NEW_DATASET_NAME]

options:
  -h, --help            show this help message and exit
  --dataset DATASET     Name of the dataset to check for contamination.
  --split SPLIT         Split to check for contamination, defaults to `train`.
  --ngram_size NGRAM_SIZE
                        Size of n-grams to build, defaults to 8.
  --problem_column PROBLEM_COLUMN
                        Name of the column containing the problem (prompt).
  --cleanup           Whether to remove the contaminated rows before pushing the dataset.
  --new_dataset_name NEW_DATASET_NAME
                        New name for the dataset. If not provided, will reuse the name and add a `_decontaminated` to the name.
```

## Evaluating models

We use `lighteval` to evaluate models, with custom tasks defined in `src/open_r1/evaluate.py`. For models which fit on a single GPU, run:

```shell
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

# AIME 2024
TASK=aime24
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# MATH-500
TASK=math_500
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# GPQA Diamond
TASK=gpqa:diamond
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# LiveCodeBench
lighteval vllm $MODEL_ARGS "extended|lcb:codegeneration|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR 
```

> [!IMPORTANT]
> You must set `max_model_length=32768` in the `vllm` command to align with the `max_new_tokens` we define per eval. Without this, `lighteval` will throw an error.

To increase throughput across multiple GPUs, use _data parallel_ as follows:

```shell
NUM_GPUS=8
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
TASK=aime24
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR 
```

For large models which require sharding across GPUs, use _tensor parallel_ and run:

```shell
NUM_GPUS=8
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
TASK=aime24
OUTPUT_DIR=data/evals/$MODEL

export VLLM_WORKER_MULTIPROC_METHOD=spawn
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR 
```

You can also launch an evaluation with `make evaluate`, specifying the model, task, and optionally the parallelism technique and number of GPUs.

To evaluate on a single GPU:

```shell
make evaluate MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B TASK=aime24
```

To use Data Parallelism:

```shell
make evaluate MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B TASK=aime24 PARALLEL=data NUM_GPUS=8
```

To use Tensor Parallelism:

```shell
make evaluate MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B TASK=aime24 PARALLEL=tensor NUM_GPUS=8
```

## Reproducing Deepseek's evaluation results

> [!NOTE]
> The DeepSeek-R1 paper uses sampling with 64 responses per query to estimate `pass@1`. Below, we report the results from sampling 1 response per query, which likely explains the small 1-3σ discrepancies between our results and theirs.

### AIME 2024

We are able to reproduce Deepseek's reported results on the AIME 2024 benchmark within ~1-3 standard deviations:

| Model                         | AIME 2024 (🤗 LightEval) | AIME 2024 (DeepSeek Reported) |
|:------------------------------|:-----------------------:|:----------------------------:|
| DeepSeek-R1-Distill-Qwen-1.5B |          26.7           |             28.9             |
| DeepSeek-R1-Distill-Qwen-7B   |          56.6           |             55.5             |
| DeepSeek-R1-Distill-Qwen-14B  |          60.0           |             69.7             |
| DeepSeek-R1-Distill-Qwen-32B  |          73.2           |             72.6             |
| DeepSeek-R1-Distill-Llama-8B  |          43.3           |             50.4             |
| DeepSeek-R1-Distill-Llama-70B |          73.3           |             70.0             |

To reproduce these results use the following command:

```shell
NUM_GPUS=1 # Set to 8 for 32B and 70B models
MODEL=deepseek-ai/{model_name}
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,data_parallel_size=$NUM_GPUS,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "custom|aime24|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
```

Alternatively, you can launch Slurm jobs as follows:

```shell
python scripts/run_benchmarks.py --model-id {model_id}  --benchmarks aime24
```

### MATH-500

We are able to reproduce Deepseek's reported results on the MATH-500 benchmark within ~1-3 standard deviations:

| Model                         | MATH-500 (🤗 LightEval) | MATH-500 (DeepSeek Reported) |
|:------------------------------|:-----------------------:|:----------------------------:|
| DeepSeek-R1-Distill-Qwen-1.5B |          84.6           |             83.9             |
| DeepSeek-R1-Distill-Qwen-7B   |          93.0           |             92.8             |
| DeepSeek-R1-Distill-Qwen-14B  |          95.0           |             93.9             |
| DeepSeek-R1-Distill-Qwen-32B  |          96.6           |             94.3             |
| DeepSeek-R1-Distill-Llama-8B  |          88.6           |             89.1             |
| DeepSeek-R1-Distill-Llama-70B |          96.4           |             94.5             |

To reproduce these results use the following command:

```shell
NUM_GPUS=1 # Set to 8 for 32B and 70B models
MODEL=deepseek-ai/{model_name}
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,data_parallel_size=$NUM_GPUS,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "custom|math_500|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
```

Alternatively, you can launch Slurm jobs as follows:

```shell
python scripts/run_benchmarks.py --model-id {model_id}  --benchmarks math_500
```

### GPQA Diamond

We are able to reproduce Deepseek's reported results on the GPQA Diamond benchmark within ~1-3 standard deviations:

| Model                         | GPQA Diamond (🤗 LightEval) | GPQA Diamond (DeepSeek Reported) |
|:------------------------------|:---------------------------:|:--------------------------------:|
| DeepSeek-R1-Distill-Qwen-1.5B |            34.3             |               33.8               |
| DeepSeek-R1-Distill-Qwen-7B   |            50.5             |               49.1               |
| DeepSeek-R1-Distill-Qwen-14B  |            59.6             |               59.1               |
| DeepSeek-R1-Distill-Qwen-32B  |            63.6             |               62.1               |
| DeepSeek-R1-Distill-Llama-8B  |            52.0             |               49.0               |
| DeepSeek-R1-Distill-Llama-70B |            67.2             |               65.2               |

To reproduce these results use the following command:

```shell
NUM_GPUS=1 # Set to 8 for 32B and 70B models
MODEL=deepseek-ai/{model_name}
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,data_parallel_size=$NUM_GPUS,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "custom|gpqa:diamond|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
```

```shell
python scripts/run_benchmarks.py --model-id {model_id}  --benchmarks gpqa
```

### LiveCodeBench

We are able to reproduce Deepseek's reported results on the LiveCodeBench code generation benchmark within ~1-3 standard deviations:

| Model                         | LiveCodeBench (🤗 LightEval) | GPQA Diamond (DeepSeek Reported) |
|:------------------------------|:----------------------------:|:--------------------------------:|
| DeepSeek-R1-Distill-Qwen-1.5B |             16.3             |               16.9               |
| DeepSeek-R1-Distill-Qwen-7B   |             36.6             |               37.6               |
| DeepSeek-R1-Distill-Qwen-14B  |             51.5             |               53.1               |
| DeepSeek-R1-Distill-Qwen-32B  |             56.6             |               57.2               |
| DeepSeek-R1-Distill-Llama-8B  |             37.0             |               39.6               |
| DeepSeek-R1-Distill-Llama-70B |             54.5             |               57.5               |

To reproduce these results use the following command:

```shell
NUM_GPUS=1 # Set to 8 for 32B and 70B models, or data_parallel_size=8 with the smaller models for speed
MODEL=deepseek-ai/{model_name}
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,data_parallel_size=$NUM_GPUS,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "extended|lcb:codegeneration|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
```

```shell
python scripts/run_benchmarks.py --model-id {model_id}  --benchmarks lcb
```

## Data generation

### Generate data from a smol distilled R1 model

The following example can be run in 1xH100. 
First install the following dependencies:

```shell
uv pip install "distilabel[vllm]>=1.5.2"
```

Now save the following snippet into a file named `pipeline.py` and run it with `python pipeline.py`. It will generate 4 outputs for each of the 10 examples (change the username for the repository to your org/user name):

```python
from datasets import load_dataset
from distilabel.models import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration


prompt_template = """\
You will be given a problem. Please reason step by step, and put your final answer within \boxed{}:
{{ instruction }}"""

dataset = load_dataset("AI-MO/NuminaMath-TIR", split="train").select(range(10))

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # Exchange with another smol distilled r1

with Pipeline(
    name="distill-qwen-7b-r1",
    description="A pipeline to generate data from a distilled r1 model",
) as pipeline:

    llm = vLLM(
        model=model_id,
        tokenizer=model_id,
        extra_kwargs={
            "tensor_parallel_size": 1,
            "max_model_len": 8192,
        },
        generation_kwargs={
            "temperature": 0.6,
            "max_new_tokens": 8192,
        },
    )
    prompt_column = "problem"
    text_generation = TextGeneration(
        llm=llm, 
        template=prompt_template,
        num_generations=4,
        input_mappings={"instruction": prompt_column} if prompt_column is not None else {}
    )


if __name__ == "__main__":
    distiset = pipeline.run(dataset=dataset)
    distiset.push_to_hub(repo_id="username/numina-deepseek-r1-qwen-7b")
```

Take a look at the sample dataset at [HuggingFaceH4/numina-deepseek-r1-qwen-7b](https://huggingface.co/datasets/HuggingFaceH4/numina-deepseek-r1-qwen-7b).

